#!/usr/bin/env python3

import argparse
import json
import logging
import os
import socket
import sys
import threading
import time
import pandas as pd
from multiprocessing import Pool
from os.path import isfile, join

VERSION = '0.3.0'

WAV_FILE = 'test_dir/wav/switchboard.wav'
WAV_DIR = 'test_dir/wav'
OUTPUT_DIR = 'test_dir/txt'
POOL_SIZE = 5

logging.basicConfig(level='INFO', format="%(levelname)s: (%(thread)d) - %(message)s")
lock = threading.Lock()

DESCRIPTION = """
The Mod9 ASR Engine TCPServer implements a custom protocol; this tool is a compliant client.

This tool reads audio data on stdin, writes server responses on stdout, and logs on stderr.
It also converts its command-line arguments into metadata/options acceptable by the server.

The input audio data is expected to be in WAV format and encoded as:
  - linear (e.g. not u-law or a-law)
  - signed (e.g. not float)
  - 16-bit (i.e. 2-byte per sample)
  - little-endian (very common)
  - mono (1 channel)
The server will check the WAV header to ensure the input audio is correctly encoded.

The audio sample rate must match the rate supported by the server's loaded acoustic model.

It is also possible to supply raw PCM (e.g. a headerless stream).  In this case, it is important
to ensure that the data is properly encoded; otherwise the ASR results will likely suffer (GIGO).
For raw format, the --rate argument must indicate the appropriate sample rate to the server.

The --command argument can be used to request a different server behavior. Currently supported:
  --command=decode           [default] Perform ASR on an input audio stream that follows.
  --command=get-stats        Report some useful server statistics, including CPU and memory usage.
  --command=get-version      Report the server version, hopefully matching this client.
  --command=shutdown         Gracefully shutdown the server (wait for pending requests to finish).
  --command=kill             Immediately stop the the server (might be restarted by daemon script).

The default command is "decode", which will perform automatic speech recognition in streaming mode,
processing the input audio stream as quickly as possible using a single CPU thread.

The server accepts processing options specified as JSON on the first line received from the client.
This client script provides a command-line interface to wrap some of these options.
Specifying --options-json may enable access to some undocumented/unsupported functionality.

The examples below illustrate some of the various ways in which this client script can interact with
the Mod9 ASR Engine TCPServer.  In addition to the curl tool (curl.haxx.se) for downloading some
test files, it can be helpful to use SoX (sox.sourceforge.net) to record and convert audio data.


EXAMPLES:

# Download a short test file (WAV) and pass it into the client with default options:
curl -sL rmtg.co/heyjohn.wav |\
 ./{scriptname} $HOST $PORT

# Use audio from an ASR eval (downmixed to mono), processing large chunks for efficiency:
curl -sL rmtg.co/switchboard.wav |\
  ./{scriptname} $HOST $PORT --latency=3

# Simulate real-time (via bitrate throttling), operating in low-latency mode, with partial results:
curl -sL rmtg.co/switchboard.wav |\
  ./{scriptname} $HOST $PORT --bitrate=128000 --latency=0.1 --partial

# Record audio from local microphone device, updating live with word-level timestamps:
sox -d -V1 -q\
 -e signed -b 16 -L -c 1 -t wav\
 -r 8000\
 - |\
 ./{scriptname} $HOST $PORT --latency=0.1 --timestamp

# Stream a live 16kHz recording from your local microphone device, as raw PCM, with formatted results:
# NOTE: the server must have 16kHz models to match this audio sample rate.
#       Generally, this will be served from port 16000.
# NOTE: the server must have the correct NLP models to format the resulting transcripts.
sox -d -V1 -q\
 -e signed -b 16 -L -c 1 -t raw\
 -r 16000\
 - |\
 ./{scriptname} $HOST $PORT -f=raw -r=16000 -tf
""".format(scriptname=os.path.basename(__file__))

# CONFIGURATION CONSTANTS
CHUNK_SIZE = 160                # enable latency as low as 0.01s at 8kHz
EOF_MARKER = 'END-OF-FILE'      # configurable sequence that's unlikely to occur in PCM audio data


def recv_response(sock, stream):
    """
    Receive JSON-formatted single-line messages from the server and write to stdout.
    We do some basic checking of the expected message format and set this thread's status.
    """
    # Use this thread object's namespace to convey this request's status to other threads.
    thread = threading.current_thread()
    thread._status = None

    # Use a file-like object, for convenience of reading lines.
    sockfile = sock.makefile(mode='r')

    while True:
        # Read a line and print it to stdout.
        try:
            line = sockfile.readline()
        except Exception:
            logging.exception('Failed to receive expected line from server.')
            sys.exit(1)
        if not line:
            break
        print(line, flush=True, end='')

        try:
            response = json.loads(line)
            if 'status' in response:
                thread._status = response['status']
        except Exception:
            logging.exception('Failed to parse the expected JSON response.')
            sys.exit(1)

def thread_routine(args_tuple):
    options_json, filename = args_tuple
    #Connect to server by initializing socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((args.host, args.port))
    except Exception:
        lock.acquire()
        logging.exception("Thread %s could not connect to %s:%s", args.host, args.port)
        lock.release()
        sys.exit(1)
    lock.acquire()
    logging.info("Connected to %s:%s", args.host, args.port)
    lock.release()

    # Start by sending the options as JSON on the first line (terminated w/ newline character).
    first_line = json.dumps(options_json, separators=(',', ':'))+'\n'
    sock.sendall(first_line.encode())

    thread_status = None

    sockfile = sock.makefile(mode='r')

    #Make sure server is ready for audio data
    while thread_status is None:
        time.sleep(1)
        try:
            line = sockfile.readline()
            # if line == "": continue
            print("LINE:", line)
            response = json.loads(line)
            if 'status' in response:
                thread_status = response['status']
        except Exception:
            lock.acquire()
            logging.exception('Failed to receive expected line from server.')
            lock.release()
            sock.close()
            return

    print("STATUS: ", thread_status)
    #If we get here, connection status is processing
    #We're free to send data
    file_stream = open(filename, "rb")
    for chunk in iter(lambda: file_stream.read(CHUNK_SIZE), b''):
        sock.sendall(chunk)

    # Send custom EOF sequence to indicate that the client is done sending, but still receiving.
    sock.sendall(options_json['eof'].encode())
    lock.acquire()
    logging.info("Done sending data.  Sent final EOF bytes: %s", options_json['eof'])
    lock.release()

    #Open file descriptor to output file
    name = filename.split("/")[-1].split(".")[0] + ".txt"
    #outfile_name = join(OUTPUT_DIR, name)
    #print('writing output to: ', outfile_name)
    #outfile = open(outfile_name, "a")
    output_text = []

    while True:
        # Read a line and print it to stdout.
        try:
            line = sockfile.readline()
        except Exception:
            lock.acquire()
            logging.exception('Failed to receive expected line from server.')
            lock.release()
            sock.close()
            #outfile.close()
            return

        if not line:
            break
        # print(line, flush=True, end='')


        try:
            response = json.loads(line)
            if 'transcript' in response:
                #print(response['transcript'])
                #outfile.write(response['transcript'] + " ")
                output_text.append(response['transcript'])
            if 'status' in response:
                thread_status = response['status']
        except Exception:
            lock.acquire()
            logging.exception('Failed to parse the expected JSON response.')
            lock.release()
            sock.close()
            #outfile.close()
            return

    lock.acquire()
    logging.info('Done receiving response.')
    lock.release()

    # Fully shutdown and close the socket.
    #outfile.close()
    sock.close()
    lock.acquire()
    logging.info('Closed connection.')
    lock.release()
    return filename, " ".join(output_text)


# ArgumentDefaultsHelpFormatter and RawDescriptionHelpFormatter override different methods
# of the base HelpFormatter class, so inheritance gives us the features of both.
class RawDescriptionWithDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                                              argparse.RawDescriptionHelpFormatter):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=RawDescriptionWithDefaultsHelpFormatter,
                                     usage='Run with the --help command line option for help.')
    parser.add_argument('host', nargs='?', default='localhost',
                        help='Server hostname.')
    parser.add_argument('port', nargs='?', type=int, default=9900,
                        help='Server TCP port.')
    parser.add_argument('--command', '-c', default='decode',
                        help='Command to send to server.')
    parser.add_argument('--format', '-f', dest='audio_format', default='wav',
                        choices=['wav', 'raw'],
                        help='Specify the input audio format.')
    parser.add_argument('--rate', '-r', type=int,
                        help='For raw audio format, specify the sample frequency in Hertz.')
    parser.add_argument('--bitrate', '-br', type=int,
                        help='Simulate network transmission at this bitrate.')
    parser.add_argument('--latency', '-l', type=float,
                        help='Desired real-time latency, in seconds.')
    parser.add_argument('--partial', '-p', action='store_true',
                        help='Non-final results are updated if words in the transcript change.')
    parser.add_argument('--confidence', action='store_true',
                        help='Output word-level confidence metrics for final results.')
    parser.add_argument('--batch', action='store_true',
                        help='Determine whether input audio is processed in batch mode.')
    parser.add_argument('--timestamp', action='store_true',
                        help='Output word-level time intervals.')
    parser.add_argument('--transcript-formatted', '-tf', action='store_true',
                        help='Output formatted transcripts for final results.')
    parser.add_argument('--options-json', '-j', type=str, default='{}',
                        help='Additional options, as JSON; will override command-line arguments.')
    parser.add_argument('--loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help='Display log messages at and above this level.')
    parser.add_argument('--version', action='store_true',
                        help='Report the version of this client (not necessarily same as server).')
    args = parser.parse_args()

    if args.version:
        print(VERSION)
        sys.exit(0)

    # Write logs to stderr.
    # Set the options specified as arguments.
    options_json = {'command': args.command}

    if args.command == 'decode':
        # Add additional args relevant for decode jobs.
        options_json['eof'] = EOF_MARKER
        if args.audio_format:
            options_json['format'] = args.audio_format
        if args.latency:
            options_json['latency'] = args.latency
        if args.partial:
            options_json['partial'] = args.partial
        if args.confidence:
            options_json['confidence'] = args.confidence
        if args.timestamp:
            options_json['timestamp'] = args.timestamp
        if args.batch:
            options_json['batch'] = args.batch
            options_json['batch-threads'] = 4 #multi-process on server side using 4 threads
        if args.transcript_formatted:
            options_json['transcript-formatted'] = args.transcript_formatted

        # Set the rate if raw format; not needed for WAV.
        if args.audio_format == 'raw':
            if args.rate is None:
                logging.error('Must specify sample rate for raw audio format.')
                sys.exit(1)
            options_json['rate'] = args.rate
        elif args.audio_format == 'wav' and args.rate is not None:
            logging.error('Audio sample rate should only be specified for raw audio format.')
            sys.exit(1)

        # Maybe override with --options-json argument.
        try:
            options_json.update(json.loads(args.options_json))
        except Exception:
            logging.exception('Could not parse --options-json argument.')
            sys.exit(1)

    all_wav_files = [(options_json, join(WAV_DIR, f)) for f in os.listdir(WAV_DIR) if isfile(join(WAV_DIR, f)) and f.split(".")[-1] == 'wav']
    threadpool = Pool(POOL_SIZE)
    output_dict = {'filename': [], 'mod9_transcription': []}
    for result in threadpool.map(thread_routine, all_wav_files):
        filename, text = result
        output_dict['filename'].append(filename.split('/')[-1])
        output_dict['mod9_transcription'].append(text)

    print("Done processing audio files...")
    output_dataframe = pd.DataFrame(output_dict)
    output_dataframe.to_csv("test_dir/test.csv", index = False)

    # Connect to the server and use a file-like interface to the socket.
    # try:
    #     sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     sock.connect((args.host, args.port))
    # except Exception:
    #     logging.exception("Could not connect to %s:%s", args.host, args.port)
    #     sys.exit(1)
    # logging.info("Connected to %s:%s", args.host, args.port)

    # Start by sending the options as JSON on the first line (terminated w/ newline character).
    # first_line = json.dumps(options_json, separators=(',', ':'))+'\n'
    # sock.sendall(first_line.encode())
    # logging.info("Sent client options as JSON on first line: %s", first_line.strip())

    # Create a separate thread to read from the socket asynchronously.
    # recv_thread = threading.Thread(target=recv_response, args=(sock, sys.stdout))
    # recv_thread.daemon = True
    # recv_thread.start()

    # Commands other than decode don't require additional data to be sent.
    #Not needed because we are only decoding
    # if args.command != 'decode':
    #     recv_thread.join()
    #     sock.close()
    #     if recv_thread._status == 'failed':
    #         sys.exit(1)
    #     else:
    #         sys.exit(0)

    # Wait until the server reports an expected initial status of 'processing'.
    # while recv_thread.isAlive() and getattr(recv_thread, '_status', None) is None:
    #     time.sleep(1)
    # if recv_thread._status != 'processing':
    #     logging.error('Expected initial "processing" status for decode command.')
    #     sock.close()
    #     sys.exit(1)

    # Send data in binary chunks, until EOF on stdin.
    # logging.info('Start sending data.')
    # start_time = time.time()
    # bits_sent = 0
    # file_stream = open(WAV_FILE, "rb")
    # for chunk in iter(lambda: file_stream.read(CHUNK_SIZE), b''):
    #     sock.sendall(chunk)
    #
    # # Send custom EOF sequence to indicate that the client is done sending, but still receiving.
    # sock.sendall(options_json['eof'].encode())
    # logging.info("Done sending data.  Sent final EOF bytes: %s", options_json['eof'])

    # Wait until the server finishes sending its response messages.
    # A blocking .join() is simpler, but main thread won't catch an interrupt (e.g. Ctl-C).
    # while recv_thread.isAlive():
    #     time.sleep(0.1)
    # logging.info('Done receiving response.')
    #
    # # Fully shutdown and close the socket.
    # sock.close()
    # logging.info('Closed connection.')

    # Exit with an appropriate exit code.
    # if recv_thread._status == 'completed':
    #     sys.exit(0)
    # elif recv_thread._status == 'failed':
    #     sys.exit(1)
    # else:
    #     logging.error("Final message status (%s) was unexpected.", recv_thread._status)
    #     sys.exit(1)
