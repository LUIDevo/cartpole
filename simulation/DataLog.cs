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

	private static int _tickInEpisode = 0;
	private static int _episodesDone   = 0;
	public static long TotalRows { get; private set; } = 0;

	public static bool EpisodeDone => _tickInEpisode >= _ticksPerEpisode;
	public static bool AllDone     => _episodesDone   >= _targetEpisodes;

	// Called once per process (guarded); re-invocations after scene reloads no-op.
	public static void Init(string outPath, int ticksPerEpisode, int targetEpisodes, int? seed)
	{
		if (_initialized) return;
		_initialized = true;

		_ticksPerEpisode = Math.Max(1, ticksPerEpisode);
		_targetEpisodes  = Math.Max(1, targetEpisodes);
		Rng = seed.HasValue ? new Random(seed.Value) : new Random();

		string dir = Path.GetDirectoryName(outPath);
		if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

		_writer = new StreamWriter(outPath, append: false);
		_writer.WriteLine("cart_velocity,pole_angular_velocity,pole_angle,motor_command");
		_writer.Flush();
	}

	// One dataset row: the 3 state features + the exact command applied to that state.
	public static void WriteRow(float cartVelocity, float poleAngularVelocity, float poleAngle, float motorCommand)
	{
		if (_writer == null) return;
		_writer.WriteLine(string.Format(CultureInfo.InvariantCulture,
			"{0:R},{1:R},{2:R},{3:R}", cartVelocity, poleAngularVelocity, poleAngle, motorCommand));
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
