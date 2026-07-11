using Godot;
using System;
using System.Globalization;

public partial class SimController : Node2D
{
	private enum Mode { DataGen, Control, Watch }
	private Mode _mode;
	private bool _resetPending;
	private double _belowTime;
	private int    _watchTicks;
	private Cart   _cart;

	private static readonly float BelowHorizon = Mathf.Pi / 2f;
	private double _graceSeconds = 1.0;
	private const int WatchMaxTicks = 3000;

	public override void _Ready()
	{
		string outPath = "data/dataset.csv";
		int ticks = 500, episodes = 100, port = 0, shard = 0;
		int? seed = null;

		foreach (string arg in OS.GetCmdlineUserArgs())
		{
			if (arg.StartsWith("--out="))            outPath = arg.Substring("--out=".Length);
			else if (arg.StartsWith("--ticks="))     { if (int.TryParse(arg.Substring("--ticks=".Length),    out int t)) ticks    = t; }
			else if (arg.StartsWith("--episodes="))  { if (int.TryParse(arg.Substring("--episodes=".Length), out int e)) episodes = e; }
			else if (arg.StartsWith("--seed="))      { if (int.TryParse(arg.Substring("--seed=".Length),     out int s)) seed     = s; }
			else if (arg.StartsWith("--port="))      { if (int.TryParse(arg.Substring("--port=".Length),     out int p)) port     = p; }
			else if (arg.StartsWith("--grace="))     { if (double.TryParse(arg.Substring("--grace=".Length), NumberStyles.Float, CultureInfo.InvariantCulture, out double g)) _graceSeconds = g; }
			else if (arg.StartsWith("--shard="))     { if (int.TryParse(arg.Substring("--shard=".Length),    out int sh)) shard    = sh; }
		}

		bool headless = DisplayServer.GetName() == "headless";
		_cart = GetNodeOrNull<Cart>("CharacterBody2D");

		if (port > 0)
		{
			_mode = Mode.Control;
			if (_cart != null) _cart.ExternalControl = true;
			ControlLink.EnsureListening(port);
		}
		else if (headless)
		{
			_mode = Mode.DataGen;
			DataLog.Init(outPath, ticks, episodes, seed, shard);
			GD.Print($"Mode: DataGen -> {outPath}");
		}
		else
		{
			_mode = Mode.Watch;
			GD.Print("Mode: Watch (random policy). Pole resets when it falls.");
		}
	}

	public override void _PhysicsProcess(double delta)
	{
		switch (_mode)
		{
			case Mode.Control: StepControl(delta); break;
			case Mode.DataGen: StepDataGen();       break;
			case Mode.Watch:   StepWatch();         break;
		}
	}

	private void StepControl(double delta)
	{
		if (_resetPending) return;
		if (!ControlLink.Accept()) return;

		SendObs();
		if (_cart.IsTerminal()) { RequestReset(); return; }
		string reply = ControlLink.ReadLine(30000);
		if (reply == null)
		{
			ControlLink.Close();
			GetTree().Quit();
			return;
		}
		HandleReply(reply, delta);
	}

	private bool FallenPastGrace(double delta)
	{
		var (_, _, poleAngle) = _cart.Observe();
		if (Mathf.Abs(poleAngle) > BelowHorizon) _belowTime += delta;
		else _belowTime = 0.0;
		return _belowTime >= _graceSeconds;
	}

	private void SendObs()
	{
		var (cartVel, poleAngVel, poleAngle) = _cart.ObserveNormalized();
		float cartPos = _cart.NormalizedCartPos();
		float reward = _cart.Reward();
		bool done = _cart.IsTerminal();
		ControlLink.WriteLine(string.Format(CultureInfo.InvariantCulture,
			"{0:R},{1:R},{2:R},{3:R},{4:R},{5}", cartVel, poleAngVel, poleAngle, cartPos, reward, done ? 1 : 0));
	}

	private void RequestReset()
	{
		_resetPending = true;
		GetTree().CallDeferred(SceneTree.MethodName.ReloadCurrentScene);
	}

	private void HandleReply(string reply, double delta)
	{
		reply = reply.Trim();
		if (reply.Equals("R", StringComparison.OrdinalIgnoreCase))
		{
			RequestReset();
			return;
		}
		if (float.TryParse(reply, NumberStyles.Float, CultureInfo.InvariantCulture, out float u))
			_cart.ApplyCommand(Mathf.Clamp(u, -1f, 1f), delta);
	}

	private void StepDataGen()
	{
		if (_resetPending) return;

		if (!FallenPastGrace(GetPhysicsProcessDeltaTime()) && !_cart.CartAtEnd() && !DataLog.EpisodeDone) return;

		DataLog.EndEpisode();

		if (DataLog.AllDone)
		{
			GD.Print($"Data collection complete: {DataLog.TotalRows} rows.");
			DataLog.Close();
			GetTree().Quit();
			return;
		}

		_resetPending = true;
		GetTree().CallDeferred(SceneTree.MethodName.ReloadCurrentScene);
	}

	private void StepWatch()
	{
		if (_resetPending) return;
		_watchTicks++;
		if (FallenPastGrace(GetPhysicsProcessDeltaTime()) || _cart.CartAtEnd() || _watchTicks > WatchMaxTicks)
		{
			_resetPending = true;
			GetTree().CallDeferred(SceneTree.MethodName.ReloadCurrentScene);
		}
	}
}
