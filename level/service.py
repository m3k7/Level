from pycontrib.service.environment import Environment
from pycontrib.service.httptemplate import HttpTemplateRequestHandler

import tornado.web
import tornado.gen
import cv2

from level.camera import Camera
from level.sensor import OpticalLevel

class HttpFace(HttpTemplateRequestHandler):

    meta = {'template': 'dashboard.htmlt'}

    def initialize(self, env, sensor):
        HttpTemplateRequestHandler.initialize(self, env)
        self.sensor = sensor
            
    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def get(self, *args):
        
        self.set_status(200)
        if 'rectified' in self.request.arguments:
            ret, jpeg = cv2.imencode('.jpg', self.sensor.getSampleFrame())
            if ret:
                self.set_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.set_header("Content-type",  "image/png")
            ans = jpeg.tostring()
            ans = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tostring() + b'\r\n\r\n'
            ans = jpeg.tostring()
            self.write(ans)
        elif 'mjpeg' in self.request.arguments:
            self.set_header('Cache-Control', 'no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0')
            self.set_header('Connection', 'keep-alive')
            boundary = '--boundarydonotcross'
            self.set_header('Content-Type', 'multipart/x-mixed-replace;boundary={0}'.format(boundary))
            self.set_header('Pragma', 'no-cache')
            while True:
                ret, jpeg = cv2.imencode('.jpg', self.sensor.getSampleFrame())
                if not ret:
                    continue
                jpeg = jpeg.tostring()
                self.write('{0}/n'.format(boundary))
                self.write('Content-type: image/jpeg\r\n')
                self.write('Content-length: {0}\r\n\r\n'.format(len(jpeg)))
                self.write(jpeg)
                yield tornado.gen.Task(self.flush)
                yield tornado.gen.sleep(0.5)
        else:
            ans = HttpFace._indext.generate(handler=self, sensor=self.sensor)
            self.write(ans)

        self.finish()
        
    @tornado.gen.coroutine
    def post(self, *args):
        
        self.set_status(200)
        args = dict()
        for arg in ('name', 'width', 'upper_shift', 'x_shift', 'lower_shift', 'password'):
            if not arg in self.request.arguments:
                self.write('Bad request')
                self.finish()
                return
            val = self.request.arguments[arg][0].decode()
            if arg in ('width', 'upper_shift', 'lower_shift') and int(val) < 0:
                self.write('Ошибка. Отрицательное значение неприменимо.')
                self.finish()
                return
            args[arg] = val
        if args['password'] != '123':
                self.write('Неверный пароль')
                self.finish()
                return
        
        self.sensor.setRectCrop(int(args['upper_shift']), int(args['lower_shift']), int(args['x_shift']), int(args['width']))
        self.write('Данные приняты')
        self.finish()
        

if __name__ == '__main__':
    env = Environment()
        
    sensor = OpticalLevel(**env.config['SENSOR'])
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
