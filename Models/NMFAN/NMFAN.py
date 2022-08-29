import matlab.engine

eng = matlab.engine.start_matlab()
eng.simple_script(nargout=0)
eng.quit()