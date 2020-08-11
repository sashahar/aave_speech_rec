#!/usr/bin/env python3

import argparse
import concurrent.futures
import csv
import json
import logging
import os
import socket
import subprocess
import sys
import threading

EOF = 'END-OF-FILE'


def listdir_abs(dirname):
    """https://stackoverflow.com/a/9816863"""
    for dirpath, _, filenames in os.walk(dirname):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def _connect_engine(host, port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
    except Exception as e:
        logging.exception("Could not connect to engine server at %s:%s", host, port)
        raise e
    return sock


def _get_version(sock):
    version_command = json.dumps({"command": "get-version"}) + '\n'
    try:
        sock.sendall(version_command.encode())
        sockfile = sock.makefile(mode='r')
        resp = sockfile.readline()
        version = json.loads(resp)['version']
    except Exception as e:
        logging.exception("Unable to get engine version. Exception: %s", e)
        raise e
    return version


def _recv_response(sock, audiofn, outfn):
    sockfile = sock.makefile(mode='r')
    while True:
        try:
            line = sockfile.readline()
        except Exception as e:
            logging.exception("Failed to receive data from the engine: %s", str(e))

        if not line:
            break

        resp = json.loads(line)
        if 'final' in resp:
            resp['segment_filename'] = audiofn  # NB: Not in Mod9 results, but is useful metadata.
            with open(outfn, 'a', newline='') as f:
                f.write(json.dumps(resp) + '\n')
            logging.debug("%s: %s", audiofn, resp['transcript'])


def _send_audio(host, port, audiofn, outfn):
    sock = _connect_engine(host, port)
    logging.debug("Connected to engine for audio %s", audiofn)
    decode_command = json.dumps({
        "command": "recognize",
        "endpoint": False,
        "format": "raw",
        "rate": 16000,
        "transcript-formatted": True,
        "cats": True,
    }) + '\n'

    # Spawn a thread for receiving data from the engine.
    recv_thread = threading.Thread(target=_recv_response, args=(sock, audiofn, outfn))
    recv_thread.start()

    # Send audio to engine.
    try:
        sock.sendall(decode_command.encode())
        # Convert to 16-bit PCM encoded WAVE file.
        ffmpeg_args = ['ffmpeg', '-hide_banner',
                       '-loglevel', 'error',
                       '-i', audiofn,
                       '-ar', '16000',
                       '-ac', '1',
                       '-c:a', 'pcm_s16le',
                       '-f', 's16le',
                       '-']
        sockfile = sock.makefile(mode='w')
        proc = subprocess.Popen(ffmpeg_args, stdout=sockfile)
        _, err = proc.communicate(timeout=5*60)  # Timeout after 5 minutes
        if proc.returncode > 0:
            logging.exception("ffmpeg subprocess returned nonzero code: %d", proc.returncode)
            raise Exception(err)
        sock.sendall(EOF.encode())
        logging.debug("Done sending data for audio file %s", audiofn)

        # Wait for recv thread to finish.
        recv_thread.join()
    finally:
        sock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', nargs='?', default='localhost',
                        help='Engine server hostname.')
    parser.add_argument('--port', '-p', nargs='?', type=int, default=16000,
                        help='Engine server port.')
    parser.add_argument('--loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Display log messages at and above this level.')
    parser.add_argument('--num-threads', type=int, default=1,
                        help='Number of concurrent connections to make to the engine server.')
    parser.add_argument('manifest_file', type=str,
                        help='Manifest file.')
    parser.add_argument('output_filename', nargs='?', type=str, default='results.jsons',
                        help='Output jsons filename.')

    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel, format="%(levelname)s: %(message)s")

    # Connect to engine server. Get version to check that it's ready to receive clients.
    sock = _connect_engine(args.host, args.port)
    try:
        version = _get_version(sock)
    except Exception:
        sock.close()
        sys.exit(1)
    sock.close()
    logging.info("Engine version: %s", version)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        with open(args.manifest_file, 'r') as f:
            reader = csv.reader(f)
            future_audiofn_map = {executor.submit(_send_audio,
                                                  args.host,
                                                  args.port,
                                                  row[0],
                                                  args.output_filename): row[0]
                                  for row in reader}
            for future in concurrent.futures.as_completed(future_audiofn_map):
                audio_fn = future_audiofn_map[future]
                try:
                    resp = future.result()
                except Exception as e:
                    logging.exception("Exception in engine processing %s: %s", audio_fn, str(e))
                    sys.exit(1)
                else:
                    logging.debug("YAHOO!")
