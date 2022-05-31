import numpy as np

def quat2mat(w, x, y, z):
        """
        Args:  4 quarternion coefficients
        Return: Corresponing 3x3 Rotation matrix 
        """
        ww, wx, wy, wz = w*w, w*x, w*y, w*z
        xx, xy, xz = x*x, x*y, x*z
        yy, yz = y*y, y*z
        zz = z*z

        n = ww + xx + yy + zz

        s = 0 if n < 1e-8 else 2 / n
        
        R = np.array([1 - s*(yy+zz),  s*(xy-wz)   ,  s*(xz+wy), 
                      s*(xy+wz)    ,  1 - s*(xx+zz), s*(yz-wx),
                      s*(xz-wy),      s*(yz+wx),     1-s*(xx+yy)]).reshape([3,3])

        return R