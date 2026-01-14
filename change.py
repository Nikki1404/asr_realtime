(venv) PS C:\Users\re_nikitav\Desktop\cx_asr_realtime\scripts> python .\ws_client.py --url ws://127.0.0.1:8002/ws/asr
Traceback (most recent call last):
  File "C:\Users\re_nikitav\Desktop\cx_asr_realtime\scripts\ws_client.py", line 201, in <module>
    asyncio.run(main())
    ~~~~~~~~~~~^^^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\runners.py", line 195, in run
    return runner.run(main)
           ~~~~~~~~~~^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\base_events.py", line 725, in run_until_complete
    return future.result()
           ~~~~~~~~~~~~~^^
  File "C:\Users\re_nikitav\Desktop\cx_asr_realtime\scripts\ws_client.py", line 165, in main
    async with websockets.connect(args.url, max_size=None) as ws:
               ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\re_nikitav\AppData\Roaming\Python\Python313\site-packages\websockets\asyncio\client.py", line 587, in __aenter__
    return await self
           ^^^^^^^^^^
  File "C:\Users\re_nikitav\AppData\Roaming\Python\Python313\site-packages\websockets\asyncio\client.py", line 541, in __await_impl__
    self.connection = await self.create_connection()
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\re_nikitav\AppData\Roaming\Python\Python313\site-packages\websockets\asyncio\client.py", line 467, in create_connection
    _, connection = await loop.create_connection(factory, **kwargs)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\base_events.py", line 1166, in create_connection
    raise exceptions[0]
  File "C:\Program Files\Python313\Lib\asyncio\base_events.py", line 1141, in create_connection
    sock = await self._connect_sock(
           ^^^^^^^^^^^^^^^^^^^^^^^^^
        exceptions, addrinfo, laddr_infos)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\base_events.py", line 1044, in _connect_sock
    await self.sock_connect(sock, address)
  File "C:\Program Files\Python313\Lib\asyncio\proactor_events.py", line 726, in sock_connect
    return await self._proactor.connect(sock, address)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\windows_events.py", line 804, in _poll
    value = callback(transferred, key, ov)
  File "C:\Program Files\Python313\Lib\asyncio\windows_events.py", line 600, in finish_connect
    ov.getresult()
    ~~~~~~~~~~~~^^
ConnectionRefusedError: [WinError 1225] The remote computer refused the network connection
(venv) PS C:\Users\re_nikitav\Desktop\cx_asr_realtime\scripts> python .\ws_client.py --url ws://127.0.0.1:8002/ws/asr
Traceback (most recent call last):
  File "C:\Users\re_nikitav\Desktop\cx_asr_realtime\scripts\ws_client.py", line 201, in <module>
    asyncio.run(main())
    ~~~~~~~~~~~^^^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\runners.py", line 195, in run
    return runner.run(main)
           ~~~~~~~~~~^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\base_events.py", line 725, in run_until_complete
    return future.result()
           ~~~~~~~~~~~~~^^
  File "C:\Users\re_nikitav\Desktop\cx_asr_realtime\scripts\ws_client.py", line 165, in main
    async with websockets.connect(args.url, max_size=None) as ws:
               ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\re_nikitav\AppData\Roaming\Python\Python313\site-packages\websockets\asyncio\client.py", line 587, in __aenter__
    return await self
           ^^^^^^^^^^
  File "C:\Users\re_nikitav\AppData\Roaming\Python\Python313\site-packages\websockets\asyncio\client.py", line 541, in __await_impl__
    self.connection = await self.create_connection()
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\re_nikitav\AppData\Roaming\Python\Python313\site-packages\websockets\asyncio\client.py", line 467, in create_connection
    _, connection = await loop.create_connection(factory, **kwargs)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\base_events.py", line 1166, in create_connection
    raise exceptions[0]
  File "C:\Program Files\Python313\Lib\asyncio\base_events.py", line 1141, in create_connection
    sock = await self._connect_sock(
           ^^^^^^^^^^^^^^^^^^^^^^^^^
        exceptions, addrinfo, laddr_infos)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\base_events.py", line 1044, in _connect_sock
    await self.sock_connect(sock, address)
  File "C:\Program Files\Python313\Lib\asyncio\proactor_events.py", line 726, in sock_connect
    return await self._proactor.connect(sock, address)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\windows_events.py", line 804, in _poll
    value = callback(transferred, key, ov)
  File "C:\Program Files\Python313\Lib\asyncio\windows_events.py", line 600, in finish_connect
    ov.getresult()
    ~~~~~~~~~~~~^^
ConnectionRefusedError: [WinError 1225] The remote computer refused the network connection
