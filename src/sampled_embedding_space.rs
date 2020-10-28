use ndarray::*;
use ndarray_linalg::*;

type ModelKey = usize;

pub struct SampledEmbeddingSpace {
    pub in_dimensions : usize,
    pub feature_dimensions : usize,
    pub out_dimensions : usize,
    pub feature_collections : Rc<[EnumFeatureCollection; 3]>,
    models : HashMap<ModelKey, Array1<f32>>,
    func_sketcher : LinearSketch
}
