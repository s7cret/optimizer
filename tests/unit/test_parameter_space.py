from optimizer import Parameter, ParameterSpace

def test_grid_and_constraints_decimal():
    s=ParameterSpace([Parameter('a','float',0.1,0.1,0.3,0.1), Parameter('b','int',1,1,3,1)], ['a < 0.31 and b >= 2'])
    grid=list(s.generate_grid())
    assert len(grid)==6
    assert grid[0]['a']==0.1
    assert s.grid_size()==9

def test_neighbors():
    s=ParameterSpace([Parameter('a','int',2,1,3,1)])
    assert len(s.neighbors({'a':2}))==2
