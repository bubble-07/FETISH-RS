use ndarray::*;
use ndarray_linalg::*;
use crate::type_id::*;
use crate::schmear::*;
use crate::func_schmear::*;
use crate::func_scatter_tensor::*;
use crate::space_info::*;
use crate::normal_inverse_wishart::*;
use crate::data_point::*;
use crate::func_schmear::*;
use crate::sigma_points::*;
use crate::model::*;
use std::collections::HashMap;

//Learned "opposite" of the sketcher for a given type

type ModelKey = usize;

pub struct Elaborator {
    pub type_id : TypeId,
    pub model : NormalInverseWishart, //Model is from projected vectors to orthog basis of kernel space of projection
    pub updates : HashMap::<ModelKey, Vec<DataPoint>>
}

impl Elaborator {
    //Before calling, need to check that there is a sketcher, and it has a kernel.
    //There's no point in constructing one of these otherwise
    pub fn new(type_id : TypeId) -> Elaborator {
        let feature_space_info = get_feature_space_info(type_id);
        let sketcher = &feature_space_info.sketcher.as_ref().unwrap();
        let sketched_dimension = sketcher.get_output_dimension();
        let kernel_mat = sketcher.get_kernel_matrix().as_ref().unwrap();
        let kernel_basis_dimension = kernel_mat.shape()[1];

        //TODO: Prior params!
        let model = NormalInverseWishart::from_in_out_dims(sketched_dimension, kernel_basis_dimension);

        Elaborator {
            type_id,
            model,
            updates : HashMap::new()
        }
    }

    pub fn expand_schmear(&self, compressed_schmear : &Schmear) -> Schmear {
        let expansion_func_schmear = self.get_expansion_func_schmear();
        expansion_func_schmear.apply(compressed_schmear)
    }

    pub fn get_expansion_func_schmear(&self) -> FuncSchmear {
        let feature_space_info = get_feature_space_info(self.type_id);
        let sketcher = &feature_space_info.sketcher.as_ref().unwrap();
        let expansion_mat = sketcher.get_expansion_matrix();

        let kernel_mat = sketcher.get_kernel_matrix().as_ref().unwrap();
        let kernel_mat_t_temp = kernel_mat.t();
        let kernel_mat_t = kernel_mat_t_temp.as_standard_layout();

        //dims: sketched_dimension -> kernel_basis_dimension
        let model_func_schmear = self.model.get_schmear();
        let model_mean = &model_func_schmear.mean;
        let model_out_covariance = &model_func_schmear.covariance.out_scatter;

        //We need to compute a func schmear of dims:
        //sketched_dimension -> full_dimension
        //As the sum of the usual pseudoinverse of the projection
        //plus the appropriately-transformed version of self.model's schmear
        let result_mean = expansion_mat + &kernel_mat.dot(model_mean);

        let result_out_covariance = kernel_mat.dot(model_out_covariance).dot(&kernel_mat_t);

        let result_covariance = FuncScatterTensor {
            in_scatter : model_func_schmear.covariance.in_scatter,
            out_scatter : result_out_covariance
        };
        
        let result_schmear = FuncSchmear {
            mean : result_mean,
            covariance : result_covariance
        };
        result_schmear
    }

    pub fn has_data(&self, update_key : &ModelKey) -> bool {
        self.updates.contains_key(update_key)
    }
    pub fn update_data(&mut self, update_key : ModelKey, data_update : &Model) {
        let feature_space_info = get_feature_space_info(self.type_id);
        let sketcher = &feature_space_info.sketcher.as_ref().unwrap();
        let kernel_mat = &sketcher.get_kernel_matrix().as_ref().unwrap();

        let func_schmear = data_update.get_schmear();        
        let schmear = func_schmear.flatten();

        let mut data_updates = Vec::new();

        let mut sigma_points = get_sigma_points(&schmear);
        let num_sigma_points = sigma_points.len();
        let weight = 1.0f32 / (num_sigma_points as f32);

        for sigma_point in sigma_points.drain(..) {
            let sketched = sketcher.sketch(&sigma_point);
            let expanded = sketcher.expand(&sketched);
            let diff = sigma_point - &expanded;
            let diff_in_kernel_basis = kernel_mat.t().dot(&diff);

            let data_update = DataPoint {
                in_vec : sketched,
                out_vec : diff_in_kernel_basis,
                weight
            };

            self.model += &data_update;

            data_updates.push(data_update);
        }

        self.updates.insert(update_key, data_updates);
    }
    pub fn downdate_data(&mut self, update_key : &ModelKey) {
        let mut data_updates = self.updates.remove(update_key).unwrap();
        for data_update in data_updates.drain(..) {
            self.model -= &data_update;
        }
    }
}
