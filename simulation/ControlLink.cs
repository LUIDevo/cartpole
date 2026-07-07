using Godot;
using System.Text;

// Process-wide TCP control channel for an external policy (e.g. a neural network).
// Lockstep protocol, newline-delimited ASCII (obs are NORMALIZED to ~[-1,1],
// same scaling as the training CSV — see Cart.ObserveNormalized):
//   Godot  -> client : "cart_velocity,pole_angular_velocity,pole_angle,cart_position,done\n"
//   client -> Godot  : "<command>\n"   (float in [-1,1]),  or  "R\n" to reset the episode
//
// State is static so it survives GetTree().ReloadCurrentScene() (episode resets).
public static class ControlLink
{
	private static TcpServer _server;
	private static StreamPeerTcp _client;
	private static readonly StringBuilder _rx = new StringBuilder();

	public static bool Active { get; private set; } = false;

	public static void EnsureListening(int port)
	{
		if (_server != null) return; // already listening (persists across reloads)
		_server = new TcpServer();
		Error err = _server.Listen((ushort)port);
		if (err != Error.Ok)
		{
			GD.PushError($"ControlLink: failed to listen on port {port}: {err}");
			_server = null;
			return;
		}
		Active = true;
		GD.Print($"ControlLink: listening on port {port}");
	}

	public static bool Connected =>
		_client != null && _client.GetStatus() == StreamPeerTcp.Status.Connected;

	// Accept a pending client (single client at a time).
	public static bool Accept()
	{
		if (Connected) return true;
		if (_server != null && _server.IsConnectionAvailable())
		{
			_client = _server.TakeConnection();
			_client.SetNoDelay(true);
			_rx.Clear();
			GD.Print("ControlLink: client connected.");
			return true;
		}
		return false;
	}

	public static void WriteLine(string line)
	{
		if (!Connected) return;
		_client.PutData(Encoding.ASCII.GetBytes(line + "\n"));
	}

	// Blocking read of one '\n'-terminated line with a wall-clock timeout (ms).
	// Returns null on timeout or disconnect. Blocking is intentional: the sim runs
	// in lockstep and only advances once the policy has replied with a command.
	public static string ReadLine(int timeoutMs)
	{
		if (!Connected) return null;
		ulong deadline = Time.GetTicksMsec() + (ulong)timeoutMs;
		while (true)
		{
			for (int i = 0; i < _rx.Length; i++)
			{
				if (_rx[i] == '\n')
				{
					string line = _rx.ToString(0, i).TrimEnd('\r');
					_rx.Remove(0, i + 1);
					return line;
				}
			}

			_client.Poll();
			if (_client.GetStatus() != StreamPeerTcp.Status.Connected) return null;

			int avail = _client.GetAvailableBytes();
			if (avail > 0)
			{
				var res = _client.GetData(avail);
				if ((Error)(int)res[0] == Error.Ok)
					_rx.Append(Encoding.ASCII.GetString((byte[])res[1]));
			}
			else
			{
				if (Time.GetTicksMsec() > deadline) return null;
				OS.DelayMsec(1);
			}
		}
	}

	// Non-blocking: return one buffered line if a full '\n'-terminated line is
	// available, else null. Never stalls the frame (used for windowed/visual mode).
	public static string TryReadLine()
	{
		if (!Connected) return null;
		_client.Poll();
		int avail = _client.GetAvailableBytes();
		if (avail > 0)
		{
			var res = _client.GetData(avail);
			if ((Error)(int)res[0] == Error.Ok)
				_rx.Append(Encoding.ASCII.GetString((byte[])res[1]));
		}
		for (int i = 0; i < _rx.Length; i++)
		{
			if (_rx[i] == '\n')
			{
				string line = _rx.ToString(0, i).TrimEnd('\r');
				_rx.Remove(0, i + 1);
				return line;
			}
		}
		return null;
	}

	public static void Close()
	{
		_client?.DisconnectFromHost();
		_server?.Stop();
		_client = null;
		_server = null;
		Active = false;
	}
}
