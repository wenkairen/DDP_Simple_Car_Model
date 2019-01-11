from abstract_cost import AbstractCost
import numpy as np
#AbstractCost
class LQRCost(AbstractCost):
  def __init__(self, N, Q, R, Qf, xd=None, ud=None):
    """
    Assuming Q,R,Qf are vectors of appropriate length
    xd can be a single goal state or a matrix [N+1, n].
    If nothing provided then treated as zero
    vector
    """
    super(LQRCost, self).__init__(N)
    self.Q = Q
    self.R = R
    self.Qf = Qf
    n = self.Q.shape[0]
    if xd is None:
        self.xd = np.zeros((N+1, n))
    elif xd.shape[0] == N+1:
        self.xd = xd
    elif xd.shape[0] == n:
        self.xd = np.tile(xd, (N+1, 1))
    else:
        raise RuntimeError("Wrong shape of xd: "+str(xd.shape()))

    m = self.R.shape[0]
    if ud is None:
        self.ud = np.zeros((N, m))
    elif ud.shape[0] == N:
        self.ud = ud
    elif ud.shape[0] == m:
        self.ud = np.tile(ud, (N, 1))
    else:
        raise RuntimeError("Wrong shape of ud: "+str(ud.shape()))

  def getQxRu(self, xdiff, udiff):
      if self.Q.ndim == 1:
          Qx = self.Q*xdiff
      else:
          Qx = np.dot(self.Q, xdiff)
      if self.R.ndim == 1:
          Ru = self.R*udiff
      else:
          Ru = np.dot(self.R, udiff)
      return Qx, Ru

  def stagewise_cost(self, i, x, u, compute_grads=False):
    xdiff =  x - self.xd[i]
    udiff = u - self.ud[i]
    Qx, Ru = self.getQxRu(xdiff, udiff)
    cost = 0.5*(np.dot(xdiff,Qx) + np.dot(u,Ru))
    if not compute_grads:
        return cost
    out = [cost]
    if compute_grads:
        # Jacobian
        out.append((Qx, Ru))
        # Hessian
        if self.Q.ndim == 1:
            Q = np.diag(self.Q)
        else:
            Q = self.Q
        if self.R.ndim == 1:
            R = np.diag(self.R)
        else:
            R = self.R
        out.append((Q, R, 0))
    return out

  def terminal_cost(self, xf, compute_grads=False):
    xdiff =  xf - self.xd[-1]
    if self.Qf.ndim == 1:
        Qfx = self.Qf*xdiff
    else:
        Qfx = np.dot(self.Qf, xdiff)
    cost = 0.5*np.dot(xdiff, Qfx)
    if not compute_grads:
        return cost
    out = [cost]
    if compute_grads:
        # Jacobian
        out.append(Qfx)
        # Hessian
        if self.Qf.ndim == 1:
            Qf = np.diag(self.Qf)
        else:
            Qf = self.Qf
        out.append(Qf)
    return out
