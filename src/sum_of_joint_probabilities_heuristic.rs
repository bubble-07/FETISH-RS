use ndarray::*;
use ndarray_linalg::*;
use crate::sampled_embedding_space::*;
use crate::model_space::*;
use crate::model::*;
use crate::sampled_term_embedding::*;
use crate::schmear::*;
use crate::inverse_schmear::*;
use crate::pseudoinverse::*;
use crate::func_schmear::*;
use crate::ellipsoid::*;
use crate::chi_squared_inverse_cdf::*;

pub fn sum_of_joint_probabilities_heuristic(model_space : &ModelSpace, ellipsoid : &Ellipsoid) -> f32 {
    let n = ellipsoid.dims();
    let chi_squared_value = chi_squared_inverse_cdf_for_heuristic_ci(n);
    let ellipsoid_center = ellipsoid.center();
    let ellipsoid_skew = ellipsoid.skew();
    let ellipsoid_skew_inv = pseudoinverse_h(ellipsoid_skew);
    //TODO: Check if this is indeed multiplication, or should it be division?
    let scaled_ellipsoid_skew_inv = chi_squared_value * ellipsoid_skew_inv;
    let ellipsoid_schmear = Schmear {
        mean : ellipsoid_center.clone(),
        covariance : scaled_ellipsoid_skew_inv
    };

    let mut tot = 0.0f32;
    for (model_key, model) in &model_space.models {
        let model_schmear = model.get_schmear();
        let model_schmear_flat = model_schmear.flatten();
        let joint_probability_integral = ellipsoid_schmear.joint_probability_integral(&model_schmear_flat);
        tot += joint_probability_integral;
    }
    tot
}
