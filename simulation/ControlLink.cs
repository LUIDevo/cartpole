using Godot;
using System.Text;

public static class ControlLink
{
	private static TcpServer _server;
	private static StreamPeerTcp _client;
	private static readonly StringBuilder _rx = new StringBuilder();

	public static bool Active { get; private set; } = false;

	public static void EnsureListening(int port)
	{
		if (_server != null) return;
		_server = new TcpServer();
		Error err = _server.Listen((ushort)port);
		if (err != Error.Ok)
		{
			GD.PushError($"ControlLink: failed to listen on port {port}: {err}");
			_server = null;
			return;
		}
		Active = true;
	}

	public static bool Connected =>
		_client != null && _client.GetStatus() == StreamPeerTcp.Status.Connected;

	public static bool Accept()
	{
		if (Connected) return true;
		if (_server != null && _server.IsConnectionAvailable())
		{
			_client = _server.TakeConnection();
			_client.SetNoDelay(true);
			_rx.Clear();
			return true;
		}
		return false;
	}

	public static void WriteLine(string line)
	{
		if (!Connected) return;
		_client.PutData(Encoding.ASCII.GetBytes(line + "\n"));
	}

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

	public static void Close()
	{
		_client?.DisconnectFromHost();
		_server?.Stop();
		_client = null;
		_server = null;
		Active = false;
	}
}
