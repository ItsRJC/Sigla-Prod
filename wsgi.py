
from app.main import app, detect_motion
import argparse
import threading
 
# python main.py --ip 0.0.0.0 --port 8080 <- to run this web
if __name__ == '__main__':

    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--ip", type=str, required=True,
    #     help="ip address of the device")
    # ap.add_argument("-o", "--port", type=int, required=True,
    #     help="ephemeral port number of the server (1024 to 65535)")
    # ap.add_argument("-f", "--frame-count", type=int, default=32,
    #     help="# of frames used to construct the background model")
    # args = vars(ap.parse_args())
    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_motion, args=(32,))
    t.daemon = True
    t.start()

    app.run(host='0.0.0.0', port='8080', debug=True, threaded=True, use_reloader=False)
