extern crate ndarray;
extern crate ndarray_linalg;

use std::ops;
use ndarray::*;
use crate::prior_specification::*;
use crate::input_to_schmeared_output::*;
use crate::array_utils::*;
use crate::params::*;
use crate::data_points::*;
use crate::pseudoinverse::*;
use crate::func_scatter_tensor::*;
use crate::func_inverse_schmear::*;
use crate::func_schmear::*;
use crate::schmear::*;
use crate::data_point::*;
use crate::sherman_morrison::*;
use crate::linalg_utils::*;
use crate::normal_inverse_wishart_sampler::*;
use crate::function_space_info::*;

use rand::prelude::*;


///Matrix-normal-inverse-wishart distribution representation
///for bayesian inference
#[derive(Clone)]
pub struct NormalInverseWishart {
    pub mean : Array2<f32>,
    ///This is always maintained to equal `mean.dot(&precision)`
    pub precision_u : Array2<f32>,
    pub precision : Array2<f32>,
    ///This is always maintained to equal `pseudoinverse_h(&precision)`
    pub sigma : Array2<f32>,
    pub big_v : Array2<f32>,
    pub little_v : f32,
    ///The output dimensionality
    pub t : usize,
    ///The input dimensionality
    pub s : usize
}

impl NormalInverseWishart {
    ///Manually re-computes `self.sigma` and `self.precision_u` to
    ///align with their definitions based on the other fields in `self`.
    ///This method takes cubic time, so it's not recommended to call this
    ///unless you observe issues with these fields accumulating numerical
    ///errors away from what they should be.
    pub fn recompute_derived(&mut self) {
        self.sigma = pseudoinverse_h(&self.precision);
        self.precision_u = self.mean.dot(&self.precision);
    }

    ///Gets the total dimensionality of `self.mean`.
    pub fn get_total_dims(&self) -> usize {
        self.s * self.t
    }

    ///Draws a sample from the represented MNIW distribution
    pub fn sample(&self, rng : &mut ThreadRng) -> Array2<f32> {
        let sampler = NormalInverseWishartSampler::new(&self);
        sampler.sample(rng)
    }

    ///The same as [`sample`], but the result is flattened.
    pub fn sample_as_vec(&self, rng : &mut ThreadRng) -> Array1<f32> {
        let thick = self.sample(rng);
        let total_dims = self.get_total_dims();
        thick.into_shape((total_dims,)).unwrap()
    }

    ///Returns the mean of the represented MNIW distribution, but flattened to be a vector.
    pub fn get_mean_as_vec(&self) -> ArrayView1::<f32> {
        flatten_matrix(self.mean.view())
    }
    ///Returns the mean of the represented MNIW distribution as a linear map.
    pub fn get_mean(&self) -> Array2<f32> {
        self.mean.clone()
    }
    ///Gets the [`FuncSchmear`] over linear mappings given by this MNIW distribution
    pub fn get_schmear(&self) -> FuncSchmear {
        FuncSchmear {
            mean : self.mean.clone(),
            covariance : self.get_covariance()
        }
    }
    ///Gets the [`FuncInverseSchmear`] over linear mappings given by this MNIW distribution.
    pub fn get_inverse_schmear(&self) -> FuncInverseSchmear {
        FuncInverseSchmear {
            mean : self.mean.clone(),
            precision : self.get_precision()
        }
    }
    ///Gets the precision (inverse covariance) [`FuncScatterTensor`] of this MNIW distribution.
    pub fn get_precision(&self) -> FuncScatterTensor {
        let scale = self.little_v - (self.t as f32) - 1.0f32;
        let mut out_precision = pseudoinverse_h(&self.big_v);
        out_precision *= scale;
        FuncScatterTensor {
            in_scatter : self.precision.clone(),
            out_scatter : out_precision
        }
    }
    ///Gets the covariance [`FuncScatterTensor`] of this MNIW distribution.
    pub fn get_covariance(&self) -> FuncScatterTensor {
        let scale = 1.0f32 / (self.little_v - (self.t as f32) - 1.0f32);
        let big_v_scaled = scale * &self.big_v;
        FuncScatterTensor {
            in_scatter : self.sigma.clone(),
            out_scatter : big_v_scaled
        }
    }
}

impl NormalInverseWishart {
    ///Constructs a [`NormalInverseWishart`] distribution from the given [`PriorSpecification`],
    ///the given feature dimensions, and the given output dimensions.
    pub fn from_in_out_dims(prior_specification : &dyn PriorSpecification,
                            feat_dims : usize, out_dims : usize) -> NormalInverseWishart {
        let mean : Array2<f32> = Array::zeros((out_dims, feat_dims));

        let in_precision_multiplier = prior_specification.get_in_precision_multiplier(feat_dims);
        let out_covariance_multiplier = prior_specification.get_out_covariance_multiplier(out_dims);

        let in_precision : Array2<f32> = in_precision_multiplier * Array::eye(feat_dims);
        let out_covariance : Array2<f32> = out_covariance_multiplier * Array::eye(out_dims);

        let little_v = prior_specification.get_out_pseudo_observations(out_dims);

        NormalInverseWishart::new(mean, in_precision, out_covariance, little_v)
    }
    ///Constructs a [`NormalInverseWishart`] distribution from the given [`PriorSpecification`]
    ///and the given [`FunctionSpaceInfo`].
    pub fn from_space_info(prior_specification : &dyn PriorSpecification,
                           func_space_info : &FunctionSpaceInfo) -> NormalInverseWishart {
        let feat_dims = func_space_info.get_feature_dimensions();
        let out_dims = func_space_info.get_output_dimensions();

        NormalInverseWishart::from_in_out_dims(prior_specification, feat_dims, out_dims)
    }
    ///Constructs a [`NormalInverseWishart`] distribution with the given mean, input precision,
    ///total output error covariance, and (pseudo-)observation count.
    pub fn new(mean : Array2<f32>, precision : Array2<f32>, big_v : Array2<f32>, little_v : f32) -> NormalInverseWishart {
        let precision_u : Array2<f32> = mean.dot(&precision);
        let sigma : Array2<f32> = pseudoinverse_h(&precision);
        let t = mean.shape()[0];
        let s = mean.shape()[1];

        NormalInverseWishart {
            mean,
            precision_u,
            precision,
            sigma,
            big_v,
            little_v,
            t,
            s
        }
    }
}

impl ops::BitXorAssign<()> for NormalInverseWishart {
    ///Inverts this [`NormalInverseWishart`] distribution in place 
    ///with respect to the addition operation given by the MNIW sum. This
    ///always will satisfy 
    ///`
    ///let mut other = self.clone();
    ///other ^= ();
    ///other += self;
    ///self == zero_normal_inverse_wishart(self.t, self.s)
    ///`
    fn bitxor_assign(&mut self, _rhs: ()) {
        self.precision_u *= -1.0;
        self.precision *= -1.0;
        self.sigma *= -1.0;
        self.little_v = 2.0 * (self.t as f32) - self.little_v;
        self.big_v *= -1.0;
    }
}

///Constructs the zero element with respect to the MNIW sum, of the given
///output and input dimensions, respectively.
fn zero_normal_inverse_wishart(t : usize, s : usize) -> NormalInverseWishart {
    NormalInverseWishart {
        mean: Array::zeros((t, s)),
        precision_u: Array::zeros((t, s)),
        precision: Array::zeros((t, s)),
        sigma: Array::zeros((t, s)),
        big_v: Array::zeros((t, s)),
        little_v: (t as f32),
        t,
        s
    }
}

impl NormalInverseWishart {
    ///Updates this [`NormalInverseWishart`] distribution to reflect
    ///new data-points for linear regression.
    pub fn update_datapoints(&mut self, data_points : &DataPoints) {
        let n = data_points.num_points();
        let X = &data_points.in_vecs;
        let Y = &data_points.out_vecs;

        let XTX = X.t().dot(X);
        let YTX = Y.t().dot(X);

        let mut out_precision = self.precision.clone();
        out_precision += &XTX;

        self.sigma = pseudoinverse_h(&out_precision);

        self.precision_u += &YTX;

        let out_mean = self.precision_u.dot(&self.sigma);

        let mean_diff = &out_mean - &self.mean;
        let mean_diff_t = mean_diff.t().clone();
        let mean_product = mean_diff.dot(&self.precision).dot(&mean_diff_t);
        self.big_v += &mean_product;
        
        let XT = X.t().clone();
        let BNX = out_mean.dot(&XT);
        let R = &Y.t() - &BNX;
        let RT = R.t().clone();
        let RTR = R.dot(&RT);
        self.big_v += &RTR;


        self.mean = out_mean;
        self.precision = out_precision;
        self.little_v += (n as f32);
    }
    fn update(&mut self, data_point : &DataPoint, downdate : bool) {
        let x = &data_point.in_vec;
        let x_norm_sq = x.dot(x);
        
        if (x_norm_sq < UPDATE_SQ_NORM_TRUNCATION_THRESH) {
            return;
        }

        let y = &data_point.out_vec;
        let s = if (downdate) {-1.0f32} else {1.0f32};
        let w = data_point.weight * s;

        let mut out_precision = self.precision.clone();
        sherman_morrison_update(&mut out_precision, &mut self.sigma, w, x.view());

        self.precision_u += &(w * outer(y.view(), x.view()));

        let out_mean = self.precision_u.dot(&self.sigma);

        self.little_v += w;

        let update_mean = (1.0f32 / x_norm_sq) * outer(y.view(), x.view());
        let update_precision = w * outer(x.view(), x.view());

        let initial_mean_diff = &out_mean - &self.mean;
        let update_mean_diff = &out_mean - &update_mean;

        self.big_v += &update_mean_diff.dot(&update_precision).dot(&update_mean_diff.t());
        self.big_v += &initial_mean_diff.dot(&self.precision).dot(&initial_mean_diff.t());

        if (self.big_v[[0, 0]] < 0.0f32) {
            println!("Big v became negative due to data update");
            println!("In vec: {}", &data_point.in_vec);
            println!("Out vec: {}", &data_point.out_vec);
            println!("Weight: {}", &data_point.weight);
            println!("Big v: {}", &self.big_v);
        }

        self.mean = out_mean;
        self.precision = out_precision;
    }
    fn update_input_to_schmeared_output(&mut self, update : &InputToSchmearedOutput, downdate : bool) {
        let data_point = DataPoint {
            in_vec : update.in_vec.clone(),
            out_vec : update.out_schmear.mean.clone(),
            weight : 1.0f32
        };
        if (downdate) {
            self.big_v -= &update.out_schmear.covariance;
        }
        self.update(&data_point, downdate);
        if (!downdate) {
            self.big_v += &update.out_schmear.covariance;
        }
    }
}

impl ops::AddAssign<&InputToSchmearedOutput> for NormalInverseWishart {
    ///Updates this [`NormalInverseWishart`] distribution to incorporate
    ///regression information from the given [`InputToSchmearedOutput`].
    fn add_assign(&mut self, update : &InputToSchmearedOutput) {
        self.update_input_to_schmeared_output(update, false);
    }
}

impl ops::SubAssign<&InputToSchmearedOutput> for NormalInverseWishart {
    ///Updates this [`NormalInverseWishart`] distribution to remove
    ///regression information from the given [`InputToSchmearedOutput`].
    fn sub_assign(&mut self, update : &InputToSchmearedOutput) {
        self.update_input_to_schmeared_output(update, true);
    }
}

impl ops::AddAssign<&DataPoint> for NormalInverseWishart {
    ///Updates this [`NormalInverseWishart`] distribution to incorporate
    ///regression information from the given [`DataPoint`].
    fn add_assign(&mut self, other: &DataPoint) {
        self.update(other, false)
    }
}

impl ops::SubAssign<&DataPoint> for NormalInverseWishart {
    ///Updates this [`NormalInverseWishart`] distribution to remove
    ///regression information from the given [`DataPoint`].
    fn sub_assign(&mut self, other: &DataPoint) {
        self.update(other, true)
    }
}

impl NormalInverseWishart {
    fn update_combine(&mut self, other : &NormalInverseWishart, downdate : bool) {
        let s = if (downdate) {-1.0f32} else {1.0f32};
        
        let mut other_precision = other.precision.clone();
        other_precision *= s;

        let mut other_big_v = other.big_v.clone();
        other_big_v *= s;

        let other_mean = &other.mean;
        let other_little_v = if (downdate) {(self.t as f32) * 2.0f32 - other.little_v} else {other.little_v};

        let mut other_precision_u = other.precision_u.clone();
        other_precision_u *= s;


        self.precision_u += &other_precision_u;

        let precision_out = &self.precision + &other_precision;

        self.sigma = pseudoinverse_h(&precision_out);

        let mean_out = self.precision_u.dot(&self.sigma);

        let mean_one_diff = &self.mean - &mean_out;
        let mean_two_diff = other_mean - &mean_out;
        
        let u_diff_l_u_diff_one = mean_one_diff.dot(&self.precision).dot(&mean_one_diff.t());
        let u_diff_l_u_diff_two = mean_two_diff.dot(&other_precision).dot(&mean_two_diff.t());

        self.little_v += other_little_v - (self.t as f32);
        self.precision = precision_out;
        self.mean = mean_out;

        self.big_v += &other_big_v;
        self.big_v += &u_diff_l_u_diff_one;
        self.big_v += &u_diff_l_u_diff_two;

        if (self.big_v[[0, 0]] < 0.0f32) {
            println!("Big v became negative due to prior update");
            println!("Big v: {}", &self.big_v);
            println!("Other big v: {}", &other_big_v);
        }
    }
}

impl ops::AddAssign<&NormalInverseWishart> for NormalInverseWishart {
    ///Updates this [`NormalInverseWishart`] distribution to the 
    ///MNIW-sum of it and the passed in [`NormalInverseWishart`] distribution.
    fn add_assign(&mut self, other: &NormalInverseWishart) {
        self.update_combine(other, false);
    }
}
impl ops::SubAssign<&NormalInverseWishart> for NormalInverseWishart {
    ///Updates this [`NormalInverseWishart`] distribution to the 
    ///MNIW-sum of it and the additive inverse of the passed in [`NormalInverseWishart`] distribution.
    fn sub_assign(&mut self, other : &NormalInverseWishart) {
        self.update_combine(other, true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;
    use crate::type_id::*;

    #[test]
    fn prior_updates_undo_cleanly() {
        let expected = random_model(*UNARY_VEC_FUNC_T);

        let mut model = expected.clone();
        let other = random_model(*UNARY_VEC_FUNC_T);

        model.data += &other.data;
        model.data -= &other.data;

        assert_equal_distributions_to_within(&model.data, &expected.data, 1.0f32);
    }

    #[test]
    fn test_model_convergence_noiseless() {
        let num_samps = 1000;
        let s = 5;
        let t = 4;
        let out_weight = 100.0f32;
        let mut model = standard_normal_inverse_wishart(s, t);

        let mat = random_matrix(t, s);
        for _ in 0..num_samps {
            let vec = random_vector(s);
            let out = mat.dot(&vec);

            let data_point = DataPoint {
                in_vec : vec,
                out_vec : out,
                weight : out_weight
            };

            model += &data_point;
        }

        assert_equal_matrices(&model.mean, &mat);
    }
}
