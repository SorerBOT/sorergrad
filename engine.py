import math

class Single:
    def __init__(self, data, _children=(), _operator='', label=""):
        self.data = data
        self._children = set(_children)
        self._operator = _operator
        self.label = label
        self.gradient = 0.0
        self._backpropagation = lambda: None
    def __repr__(self):
        return f"[data={self.data}]"
    def __add__(self, other):
        other = other if (isinstance(other, Single)) else Single(other)
        output = Single(self.data + other.data, (self, other), '+')

        def _backpropagation():
            self.gradient += 1.0 * output.gradient
            other.gradient += 1.0 * output.gradient
        output._backpropagation = _backpropagation
        
        return output
    def __radd__(self, other):
        return self + other
    def __mul__(self, other):
        other = other if (isinstance(other, Single)) else Single(other)
        output = Single(self.data * other.data, (self, other), '*')

        def _backpropagation():
            self.gradient += other.data * output.gradient
            other.gradient += self.data * output.gradient
        output._backpropagation = _backpropagation

        return output
    def __rmul__(self, other):
        return self * other
    def __neg__(self):
        return -1 * self
    def __sub__(self, other):
        return self + (-other)
    def __rsub__(self, other):
        return (-self) + other
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Error, only (int & float are supported)"
        output = Single(self.data ** other, (self,), f'**{other}')

        def _backpropagation():
            self.gradient += (other * self.data ** (other - 1)) * output.gradient
        output._backpropagation = _backpropagation

        return output
    def __truediv__(self, other):
        return self * (other**-1)
    def __rtruediv__(self, other):
        return (self**-1) * other
    def relu(self):
        output = Single(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backpropagation():
            self.gradient += (output.data > 0) * output.gradient
        output._backpropagation = _backpropagation

        return output
    def sigmoid(self):
        x = self.data
        y = 1 / (1 + math.exp(-x))
        output = Single(y, (self,), 'sigmoid')

        def _backpropagation():
            self.gradient += y * (1 - y) * output.gradient
        output._backpropagation = _backpropagation
        
        return output
    def tanh(self):
        x = self.data
        y = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        output = Single(y, (self, ), 'tanh')
        
        def _backpropagation():
          self.gradient += (1 - y**2) * output.gradient
        output._backpropagation = _backpropagation
    
        return output
    def exp(self):
        x = self.data
        output = Single(math.exp(x), (self,), 'exp')

        def _backpropagation():
            self.gradient += output.data * output.gradient
        output._backpropagation = _backpropagation

        return output
    # 100% copied from Andrej Karpathy's video
    def backpropagation(self):
        topo = []
        visited = set()
        def build_topo(v):
          if v not in visited:
            visited.add(v)
            for child in v._children:
              build_topo(child)
            topo.append(v)
        build_topo(self)
        
        self.gradient = 1.0
        for node in reversed(topo):
          node._backpropagation()