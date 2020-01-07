import logging
import yaml
import time
import tornado.web
import tornado.escape
import setup_log
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if os.path.exists("config.yaml"):
    with open("config.yaml", 'rt') as f:
        config = yaml.safe_load(f.read())


class BaseHandler(tornado.web.RequestHandler):
    def write_error(self, status_code, **kwargs):
        self.finish({
            'error': {
                'code': status_code,
                'message': self._reason,
            }
        })


class NotFoundHandler(BaseHandler):
    def get(self, *args, **kwargs):
        raise tornado.web.HTTPError(
            status_code=404,
            reason="Invalid resource path."
        )


class Service_check_handler(tornado.web.RequestHandler):
    def get(self):
        start = time.time()
        stop = time.time()
        logging.info("Total Response time %s seconds", (stop - start))
        self.write('PEA text classifcation service is up !')


# End points
application = tornado.web.Application(
    [(r"/check", Service_check_handler),
     (r"(.*)", NotFoundHandler)],
    debug=True)

if __name__ == "__main__":
    PORT = int(config["dev"]["Server"]["Port"])
    IP = config["dev"]["Server"]["IPAddress"]
    application.on_close(PORT)
    setup_log.setup_logging()
    application.listen(PORT, address=IP, max_buffer_size=1048576000)
    logging.info("""PEA Classification Service is Up at Port {} and IP {}!
                 """.format(PORT, IP))
    check_url = "http://{}:{}/check".format(IP, PORT)
    logging.info("""Check the Status at {}""".format(check_url))
    tornado.ioloop.IOLoop.instance().start()
