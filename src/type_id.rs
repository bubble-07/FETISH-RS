use std::collections::HashMap;
use lazy_static::*;
use crate::params::*;
use crate::type_graph::*;
use std::fmt;
use rand::prelude::*;

extern crate pretty_env_logger;

lazy_static! {
    static ref GLOBAL_TYPE_INFO : GlobalTypeInfo = {
        let mut types : GlobalTypeInfo = GlobalTypeInfo::new();
        let scalar_t = types.add(Type::VecType(1));
        let vector_t = types.add(Type::VecType(DIM));
        let unary_vec_func_t = types.add(Type::FuncType(vector_t, vector_t));
        let _binary_vec_func_t = types.add(Type::FuncType(vector_t, unary_vec_func_t));
        let unary_scalar_func_t = types.add(Type::FuncType(scalar_t, scalar_t));
        let binary_scalar_func_t = types.add(Type::FuncType(scalar_t, unary_scalar_func_t));
        let _map_func_t = types.add(Type::FuncType(unary_scalar_func_t, unary_vec_func_t));
        let vector_to_scalar_func_t = types.add(Type::FuncType(vector_t, scalar_t));
        let reduce_temp_t = types.add(Type::FuncType(scalar_t, vector_to_scalar_func_t));
        let _reduce_func_t = types.add(Type::FuncType(binary_scalar_func_t, reduce_temp_t));
        let _fill_func_t = types.add(Type::FuncType(scalar_t, vector_t));
        let scalar_to_vector_func_t = types.add(Type::FuncType(scalar_t, vector_t));
        let _set_head_func_t = types.add(Type::FuncType(vector_t, scalar_to_vector_func_t));
        //TODO: ensure that the types here are exactly closed w.r.t. all types
        //in func_impl
        
        info!("Adding composition types");
        
        //Add all composition types of vector functions
        for n in [1, DIM].iter() {
            let n_t = types.add(Type::VecType(*n));
            for m in [1, DIM].iter() {
                let m_t = types.add(Type::VecType(*m));
                for p in [1, DIM].iter() {
                    let p_t = types.add(Type::VecType(*p));

                    let func_one = types.add(Type::FuncType(m_t, p_t));
                    let func_two = types.add(Type::FuncType(n_t, m_t));
                    let func_out = types.add(Type::FuncType(n_t, p_t));

                    let two_to_out = types.add(Type::FuncType(func_two, func_out));
                    let _compose_type = types.add(Type::FuncType(func_one, two_to_out));
                }
            }
        }

        info!("Adding constant types");
        //Add in all constant functions
        for n in [1, DIM].iter() {
            let n_t = types.add(Type::VecType(*n));
            for m in [1, DIM].iter() {
                let m_t = types.add(Type::VecType(*m));
                let out_func_t = types.add(Type::FuncType(m_t, n_t));
                let _const_func_t = types.add(Type::FuncType(n_t, out_func_t));
            }
        }

        //Fill in the ret_types table
       for ret_type_id in 0..types.info_vec.len() {
            if (!types.ret_map.contains_key(&ret_type_id)) {
                types.ret_map.insert(ret_type_id, Vec::new());
            }
        } 

        for i in 0..types.info_vec.len() {
            let type_id = i as TypeId;
            if let Type::FuncType(arg_type_id, ret_type_id) = types.info_vec[type_id].clone() {
                let vec : &mut Vec::<(TypeId, TypeId)> = types.ret_map.get_mut(&ret_type_id).unwrap();
                vec.push((type_id, arg_type_id));
            }
        }
        
        info!("Type initialization complete");

        types
    };

    pub static ref SCALAR_T : TypeId = {
        GLOBAL_TYPE_INFO.get(&Type::VecType(1))
    };

    pub static ref VECTOR_T : TypeId = {
        GLOBAL_TYPE_INFO.get(&Type::VecType(DIM))
    };

    pub static ref UNARY_VEC_FUNC_T : TypeId = {
        GLOBAL_TYPE_INFO.get(&Type::FuncType(*VECTOR_T, *VECTOR_T))
    };

    pub static ref SCALAR_TO_VEC_FUNC_T : TypeId = {
        GLOBAL_TYPE_INFO.get(&Type::FuncType(*SCALAR_T, *VECTOR_T))
    };

    pub static ref VEC_TO_SCALAR_FUNC_T : TypeId = {
        GLOBAL_TYPE_INFO.get(&Type::FuncType(*VECTOR_T, *SCALAR_T))
    };
    
    pub static ref UNARY_SCALAR_FUNC_T : TypeId = {
        GLOBAL_TYPE_INFO.get(&Type::FuncType(*SCALAR_T, *SCALAR_T))
    };

    pub static ref BINARY_SCALAR_FUNC_T : TypeId = {
        GLOBAL_TYPE_INFO.get(&Type::FuncType(*SCALAR_T, *UNARY_SCALAR_FUNC_T))
    };
}

pub fn is_vector_type(id : TypeId) -> bool {
    id == *VECTOR_T || id == *SCALAR_T
}

struct GlobalTypeInfo {
    info_vec : Vec::<Type>,
    ind_map : HashMap<Type, TypeId>,
    ret_map : HashMap::<TypeId, Vec::<(TypeId, TypeId)>>
}

pub fn total_num_types() -> usize {
    GLOBAL_TYPE_INFO.info_vec.len()
}

pub fn get_random_type_id(rng : &mut ThreadRng) -> TypeId {
    let i : usize = rng.gen();
    let id = i % total_num_types();
    id as TypeId
}

pub fn get_random_func_type_id(rng : &mut ThreadRng) -> TypeId {
    loop {
        let type_id = get_random_type_id(rng);
        if (!is_vector_type(type_id)) {
            return type_id;
        }
    }
}

impl GlobalTypeInfo {
    fn new() -> Self {
        GlobalTypeInfo {
            info_vec : Vec::new(),
            ind_map : HashMap::new(),
            ret_map : HashMap::new()
        }
    }
    fn add(&mut self, info : Type) -> TypeId {
        if (self.ind_map.contains_key(&info)) {
            self.get(&info)
        } else {
            let ret_ind : usize = self.info_vec.len();
            self.ind_map.insert(info.clone(), ret_ind);
            self.info_vec.push(info); 
            ret_ind
        }
    }
    fn has_type(&self, info : &Type) -> bool {
        self.ind_map.contains_key(info)
    }
    fn get(&self, info : &Type) -> TypeId {
        self.ind_map.get(info).unwrap().clone()
    }
    fn get_type(&self, id : TypeId) -> Type {
        self.info_vec[id].clone()
    }
    fn get_application_type_ids(&self, id : TypeId) -> Vec::<(TypeId, TypeId)> {
        self.ret_map.get(&id).unwrap().clone()
    }
}

pub fn get_application_type_ids(id : TypeId) -> Vec::<(TypeId, TypeId)> {
    GLOBAL_TYPE_INFO.get_application_type_ids(id)
}
pub fn has_type(kind : &Type) -> bool {
    GLOBAL_TYPE_INFO.has_type(kind)
}

pub fn get_type_id(kind : &Type) -> TypeId {
    GLOBAL_TYPE_INFO.get(kind)
}
pub fn get_type(id : TypeId) -> Type {
    GLOBAL_TYPE_INFO.get_type(id)
}
pub fn get_arg_type_id(func_type_id : TypeId) -> TypeId {
    let func_type = get_type(func_type_id);
    if let Type::FuncType(arg_type_id, _) = func_type {
        arg_type_id
    } else {
        panic!();
    }
}
pub fn get_ret_type_id(func_type_id : TypeId) -> TypeId {
    let func_type = get_type(func_type_id);
    if let Type::FuncType(_, ret_type_id) = func_type {
        ret_type_id
    } else {
        panic!();
    }
}

pub type TypeId = usize;

#[derive(Eq, PartialEq, Hash, Debug, Clone)]
pub enum Type {
    VecType(usize),
    FuncType(TypeId, TypeId)
}

impl fmt::Display for Type {
    fn fmt(&self, f : &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self) {
            Type::VecType(n) => write!(f, "{}", n),
            Type::FuncType(arg, ret) => write!(f, "({} -> {})", get_type(*arg), get_type(*ret))
        }
    }
}
