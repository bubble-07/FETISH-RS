use ndarray::*;
use fetish_lib::everything::*;

#[derive(Clone)]
pub struct VectorApplicationResult<'a> {
    pub func_type_id : TypeId,
    pub func_vec : Array1<f32>,
    pub arg_vec : Array1<f32>,
    pub ret_vec : Array1<f32>,
    pub ctxt : &'a Context
}

impl<'a> VectorApplicationResult<'a> {
    pub fn get_func_vec(&self) -> TypedVector {
        TypedVector {
            type_id : self.func_type_id,
            vec : self.func_vec.clone()
        }
    }
    pub fn get_arg_vec(&self) -> TypedVector {
        let type_id = self.ctxt.get_arg_type_id(self.func_type_id);
        TypedVector {
            type_id,
            vec : self.arg_vec.clone()
        }
    }
    pub fn get_ret_vec(&self) -> TypedVector {
        let type_id = self.ctxt.get_ret_type_id(self.func_type_id);
        TypedVector {
            type_id,
            vec : self.ret_vec.clone()
        }
    }
}
