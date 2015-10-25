from pycontrib.service.environment import Environment
from pycontrib.service.httptemplate import HttpTemplateRequestHandler

import tornado.web
import cv2

from level.camera import Camera
from level.sensor import OpticalLevel

class HttpFace(HttpTemplateRequestHandler):

    meta = {'template': 'dashboard.htmlt'}

    def initialize(self, env, sensor):
        HttpTemplateRequestHandler.initialize(self, env)
        self.sensor = sensor
            
    @tornado.gen.coroutine
    def get(self, *args):
        
        if 'rectified' in self.request.arguments:
            ret, jpeg = cv2.imencode('.jpg', self.sensor.getSampleFrame())
#             if ret:
#             self.set_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.set_header("Content-type",  "image/png")
            ans = jpeg.tostring()
#             ans = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tostring() + b'\r\n\r\n'
#             ans = jpeg.tostring()
        else:
            ans = HttpFace._indext.generate(handler=self, sensor=self.sensor)
            
        self.set_status(200)
        
        self.write(ans)
        self.finish()
        

if __name__ == '__main__':
    env = Environment()
        
    stream = Camera(pi_camera=True) #file='/home/maxim/dev/level/records/video3.h264')
    sensor = OpticalLevel(stream)
    sensor.start()
        
    settings = {
        'cookie_secret': '234445734123451172353564537',
        'login_url': './login',
    }
#     settings.update(env.login)
    
    appMap = [
              ('/', HttpFace, dict(env=env, sensor=sensor)),
              ]
    app = tornado.web.Application(appMap, debug=env.debug, autoreload=False, **settings)    
    app.listen(env.port)
    
    tornado.ioloop.IOLoop.instance().start()
