using Godot;
using System;

// Root of env.tscn. Drives episode counting, reset, and process shutdown for
// headless dataset generation. Cart.cs writes the per-tick rows itself.
public partial class SimController : Node2D
{
	public override void _Ready()
	{
		// Config comes from cmdline args after `--`:
		//   --out=<path> --ticks=<per-episode> --episodes=<count> --seed=<int>
		string outPath = "data/dataset.csv";
		int ticks = 500, episodes = 100;
		int? seed = null;

		foreach (string arg in OS.GetCmdlineUserArgs())
		{
			if (arg.StartsWith("--out="))            outPath = arg.Substring("--out=".Length);
			else if (arg.StartsWith("--ticks="))     { if (int.TryParse(arg.Substring("--ticks=".Length),    out int t)) ticks    = t; }
			else if (arg.StartsWith("--episodes="))  { if (int.TryParse(arg.Substring("--episodes=".Length), out int e)) episodes = e; }
			else if (arg.StartsWith("--seed="))      { if (int.TryParse(arg.Substring("--seed=".Length),     out int s)) seed     = s; }
		}

		DataLog.Init(outPath, ticks, episodes, seed); // guarded — no-op after reloads
	}

	private bool _resetPending = false;

	public override void _PhysicsProcess(double delta)
	{
		if (_resetPending || !DataLog.EpisodeDone) return;

		DataLog.EndEpisode(); // count the episode just finished

		if (DataLog.AllDone)
		{
			GD.Print($"Data collection complete: {DataLog.TotalRows} rows.");
			DataLog.Close();
			GetTree().Quit();
			return;
		}

		// New episode = fresh scene => re-runs all _Ready/_IntegrateForces
		// randomization (cart, pole, motor, start state) and resets positions.
		// Must be deferred: reloading inside the physics callback is illegal.
		_resetPending = true;
		GetTree().CallDeferred(SceneTree.MethodName.ReloadCurrentScene);
	}
}
