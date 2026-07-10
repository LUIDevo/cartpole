using Godot;
using System;
using System.Globalization;

// Root of env.tscn. Picks a mode from cmdline args (after `--`) and whether a
// window exists:
//   DataGen  : headless, no --port. Cart self-drives random policy; rows -> CSV.
//              --out=<path> --ticks=<per-episode> --episodes=<count> --seed=<int>
//   Control  : --port=N. External policy drives the cart over TCP.
//              headless  -> strict blocking lockstep (deterministic; for training)
//              windowed  -> non-blocking request/response (smooth; watch it live)
//   Watch    : windowed, no --port. Cart self-drives random policy on screen,
//              resets when the pole falls. Just run it to SEE something.
public partial class SimController : Node2D
{
	private enum Mode { DataGen, Control, Watch }
	private Mode _mode;
	private bool _blocking;        // control mode: block the frame waiting for a command
	private bool _resetPending;
	private bool _awaitingReply;   // non-blocking control: obs sent, waiting for reply
	private double _belowTime; // seconds the pole has been continuously below horizontal
	private int    _watchTicks;
	private Cart   _cart;

	// Pole tip drops below the horizontal (top semicircle) when |angle| > 90°.
	// We only reset after it has stayed below that long, so brief dips that recover
	// keep their (valuable, near-upright) datapoints instead of ending the episode.
	private static readonly float BelowHorizon = Mathf.Pi / 2f;
	private double _graceSeconds = 1.0;         // must stay below this long -> reset (--grace)
	private const int WatchMaxTicks = 3000;     // safety cap for watch mode

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
			_blocking = headless; // window => don't freeze rendering
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
			_mode = Mode.Watch; // just watch the random policy on screen
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
		if (!ControlLink.Accept()) return; // idle until a client connects

		if (_blocking)
		{
			SendObs();
			if (_cart.IsTerminal()) { RequestReset(); return; } // done -> auto-restart, no reply expected
			string reply = ControlLink.ReadLine(30000);
			if (reply == null)
			{
				GD.Print("ControlLink: client disconnected — quitting.");
				ControlLink.Close();
				GetTree().Quit();
				return;
			}
			HandleReply(reply, delta);
			return;
		}

		// Non-blocking request/response: send one obs, then poll for its reply
		// across frames so the window keeps rendering. Cart coasts meanwhile.
		if (!_awaitingReply)
		{
			SendObs();
			if (_cart.IsTerminal()) { RequestReset(); return; } // done -> auto-restart, no reply expected
			_awaitingReply = true;
		}

		if (!ControlLink.Connected) { _awaitingReply = false; return; } // client gone; keep window

		string line = ControlLink.TryReadLine();
		if (line != null)
		{
			_awaitingReply = false;
			HandleReply(line, delta);
		}
	}

	// True once the pole has stayed below horizontal continuously for GraceSeconds.
	private bool FallenPastGrace(double delta)
	{
		var (_, _, poleAngle) = _cart.Observe();
		if (Mathf.Abs(poleAngle) > BelowHorizon) _belowTime += delta;
		else _belowTime = 0.0; // recovered above horizontal -> keep the episode alive
		return _belowTime >= _graceSeconds;
	}

	private void SendObs()
	{
		var (cartVel, poleAngVel, poleAngle) = _cart.ObserveNormalized(); // model-facing scale
		float cartPos = _cart.NormalizedCartPos();
		float reward = _cart.Reward();    // per-step state cost (<= 0) for the current state
		bool done = _cart.IsTerminal();   // pole past fail angle, or cart at a track end
		ControlLink.WriteLine(string.Format(CultureInfo.InvariantCulture,
			"{0:R},{1:R},{2:R},{3:R},{4:R},{5}", cartVel, poleAngVel, poleAngle, cartPos, reward, done ? 1 : 0));
	}

	// Ask the scene tree to reload (one episode = one scene). Deferred so it runs
	// after the current physics frame. _resetPending gates further stepping.
	private void RequestReset()
	{
		_resetPending = true;
		GetTree().CallDeferred(SceneTree.MethodName.ReloadCurrentScene);
	}

	private void HandleReply(string reply, double delta)
	{
		reply = reply.Trim();
		if (reply.Equals("R", StringComparison.OrdinalIgnoreCase)) // legacy client-driven reset
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

		// Episode ends when the pole has fallen below horizontal past the grace
		// period, the cart hits a track end (hard boundary, no grace), or the
		// per-episode tick cap is hit (safety bound).
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
