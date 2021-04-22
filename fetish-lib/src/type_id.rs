use std::collections::HashMap;
use crate::context::*;
use crate::params::*;
use crate::displayable_with_context::*;
use std::fmt;
use rand::prelude::*;

pub struct TypeInfoDirectory {
    info_vec : Vec::<Type>,
    func_ind_map : HashMap<(TypeId, TypeId), TypeId>,
    ret_map : HashMap::<TypeId, Vec::<(TypeId, TypeId)>>
}

impl TypeInfoDirectory {
    pub fn new() -> Self {
        TypeInfoDirectory {
            info_vec : Vec::new(),
            func_ind_map : HashMap::new(),
            ret_map : HashMap::new()
        }
    }
    pub fn add(&mut self, info : Type) -> TypeId {
        let added_type_id : usize = self.info_vec.len();

        self.ret_map.insert(added_type_id, Vec::new());

        if let Type::FuncType(arg_type, ret_type) = info {
            let pair = (arg_type, ret_type);
            if (self.func_ind_map.contains_key(&pair)) {
                return *self.func_ind_map.get(&pair).unwrap();
            } else {
                self.func_ind_map.insert(pair, added_type_id);
            }

            let ret_row = self.ret_map.get_mut(&ret_type).unwrap();
            ret_row.push((added_type_id, arg_type));
        }

        self.info_vec.push(info); 
        added_type_id
    }
    pub fn get_total_num_types(&self) -> usize {
        self.info_vec.len()
    }
    pub fn has_func_type(&self, arg_type_id : TypeId, ret_type_id : TypeId) -> bool {
        let pair = (arg_type_id, ret_type_id);
        self.func_ind_map.contains_key(&pair)
    }
    pub fn get_func_type_id(&self, arg_type_id : TypeId, ret_type_id : TypeId) -> TypeId {
        let pair = (arg_type_id, ret_type_id);
        *self.func_ind_map.get(&pair).unwrap()
    }
    pub fn get_type(&self, id : TypeId) -> Type {
        self.info_vec[id]
    }
    pub fn get_application_type_ids(&self, id : TypeId) -> Vec::<(TypeId, TypeId)> {
        self.ret_map.get(&id).unwrap().clone()
    }
    pub fn get_ret_type_id(&self, func_type_id : TypeId) -> TypeId {
        let func_type = self.get_type(func_type_id);
        if let Type::FuncType(_, ret_type_id) = func_type {
            ret_type_id
        } else {
            panic!();
        }
    }
    pub fn is_vector_type(&self, id : TypeId) -> bool {
        let kind = self.get_type(id);
        match (kind) {
            Type::FuncType(_, _) => false,
            Type::VecType(_) => true
        }
    }
    pub fn get_arg_type_id(&self, func_type_id : TypeId) -> TypeId {
        let func_type = self.get_type(func_type_id);
        if let Type::FuncType(arg_type_id, _) = func_type {
            arg_type_id
        } else {
            panic!();
        }
    }
    pub fn get_dimension(&self, vec_type_id : TypeId) -> usize {
        let vec_type = self.get_type(vec_type_id);
        if let Type::VecType(dim) = vec_type {
            dim
        } else {
            panic!();
        }
    }
}

pub type TypeId = usize;

#[derive(Copy, Eq, PartialEq, Hash, Debug, Clone)]
pub enum Type {
    VecType(usize),
    FuncType(TypeId, TypeId)
}

impl DisplayableWithContext for Type {
    fn display(&self, ctxt : &Context) -> String {
        match (self) {
            Type::VecType(n) => format!("{}", n),
            Type::FuncType(arg, ret) => format!("({} -> {})", 
                                        ctxt.get_type(*arg).display(ctxt), 
                                        ctxt.get_type(*ret).display(ctxt))
        }
    }
}
