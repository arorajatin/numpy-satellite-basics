import numpy as np

# earth's radius
R_E = 6378.137

rng = np.random.default_rng(42)

def gen():
  # 1000 samples from the [LEO, GEO]
  radii = rng.uniform(low=R_E + 200,high=42164,size=1000)
  # predict: shape: (1000,), min: above 6578, max: below: 42164
  assert radii.min() > 6578
  assert radii.max() < 42164
  assert radii.shape == (1000,)

  # step 3: get the direction vectors (x, y, z) X 1000
  vecs = rng.normal(size=(1000, 3))
  # axis = 1 means squash each row
  # axis = 0 means squash each col
  lengths = np.linalg.norm(vecs, axis=1, keepdims=True)
  units = vecs / lengths
  
  # get altitude
  # position = direction × distance; radii[:, None] makes (1000,) → (1000, 1) so it broadcasts across x, y, z
  positions = radii[:, None] * units 

  assert np.allclose(np.linalg.norm(units, axis=1), 1)
  assert positions.shape == (1000, 3)

  # predict - now when 1000, 1 and 1000, 3 are multiplied, that will result in 1000, 3
  #print(positions)  
    
  # now let's compute altitudes, i.e. distance from earth's surface
  # to do so, we need to deduct positions by R_E
  # compress across x axis
  # keepdims should be false because the want the altitudes do not need to be arrays 
  altitudes = np.linalg.norm(positions, axis=1) - R_E

  # round-trip check: radius → position → altitude must match radius - R_E (allclose tolerates float rounding)
  assert np.allclose(altitudes, radii - R_E)
 
  print(altitudes.min(), altitudes.max(), altitudes.mean(), positions.dtype) 

gen()
