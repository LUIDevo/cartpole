using Godot;
using System;
using System.IO;
using System.Globalization;

// Process-wide data-collection sink. Static state survives ReloadCurrentScene()
// (used to reset each episode) because statics persist for the whole Godot run.
public static class DataLog
{
	// Single shared RNG stream so all randomization (cart, pole, motor, policy)
	// is reproducible from one --seed and differs per shard.
	public static Random Rng { get; private set; } = new Random();

	private static StreamWriter _writer;
	private static bool _initialized = false;

	private static int _ticksPerEpisode = 500;
	private static int _targetEpisodes  = 100;
	private static int _shard = 0;

	private static int _tickInEpisode = 0; // step index within the current episode
	private static int _episodesDone   = 0;
	public static long TotalRows { get; private set; } = 0;

	public static bool EpisodeDone => _tickInEpisode >= _ticksPerEpisode;
	public static bool AllDone     => _episodesDone   >= _targetEpisodes;

	// True on the final row of the episode's tick budget (the step-limit condition
	// for the `done` flag). WriteRow increments after logging, so the last logged
	// row is at index _ticksPerEpisode - 1.
	public static bool IsLastStep => _tickInEpisode >= _ticksPerEpisode - 1;

	// Globally-unique batch id: one episode = one fully-randomized system. Shard is
	// folded in so ids never collide when shard CSVs are merged.
	private static long EpisodeId => (long)_shard * 1_000_000L + _episodesDone;

	// Called once per process (guarded); re-invocations after scene reloads no-op.
	public static void Init(string outPath, int ticksPerEpisode, int targetEpisodes, int? seed, int shard)
	{
		if (_initialized) return;
		_initialized = true;

		_ticksPerEpisode = Math.Max(1, ticksPerEpisode);
		_targetEpisodes  = Math.Max(1, targetEpisodes);
		_shard = shard;
		Rng = seed.HasValue ? new Random(seed.Value) : new Random();

		string dir = Path.GetDirectoryName(outPath);
		if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

		_writer = new StreamWriter(outPath, append: false);
		_writer.WriteLine("episode_id,step,cart_velocity,pole_angular_velocity,pole_angle,cart_position,motor_command,reward,done");
		_writer.Flush();
	}

	// One dataset row: episode batch id + step, then the 4 state features, the
	// exact command applied to that state, the per-step reward, and the terminal flag.
	public static void WriteRow(float cartVelocity, float poleAngularVelocity, float poleAngle, float cartPosition, float motorCommand, float reward, bool done)
	{
		if (_writer == null) return;
		_writer.WriteLine(string.Format(CultureInfo.InvariantCulture,
			"{0},{1},{2:R},{3:R},{4:R},{5:R},{6:R},{7:R},{8}",
			EpisodeId, _tickInEpisode, cartVelocity, poleAngularVelocity, poleAngle, cartPosition, motorCommand, reward, done ? 1 : 0));
		_tickInEpisode++;
		TotalRows++;
	}

	public static void EndEpisode()
	{
		_episodesDone++;
		_tickInEpisode = 0;
		_writer?.Flush();
	}

	public static void Close()
	{
		_writer?.Flush();
		_writer?.Dispose();
		_writer = null;
	}
}
