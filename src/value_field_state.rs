use crate::params::*;
use crate::schmeared_hole::*;
use crate::constraint_collection::*;
use std::collections::HashMap;
use crate::vector_application_result::*;
use crate::type_id::*;
use crate::space_info::*;
use crate::value_field::*;
use crate::sampled_embedder_state::*;
use crate::typed_vector::*;
use crate::sampled_value_field_state::*;

pub struct ValueFieldState {
    pub value_fields : HashMap<TypeId, ValueField>
}

impl ValueFieldState {
    pub fn new(target : SchmearedHole) -> ValueFieldState {
        let mut value_fields = HashMap::new();
        for func_type_id in 0..total_num_types() {
            if (!is_vector_type(func_type_id)) {
                let value_field = if (func_type_id == target.type_id) {
                                      //If the target, construct from the target
                                      ValueField::new(target.clone())
                                  } else {
                                      ValueField::from_type_id(func_type_id)
                                  };
                value_fields.insert(func_type_id, value_field);
            }
        }
        ValueFieldState {
            value_fields
        }
    }
    pub fn sample(&self, sampled_embedder_state : &SampledEmbedderState) -> SampledValueFieldState {
        let mut sampled_value_fields = HashMap::new();
        for (type_id, value_field) in self.value_fields.iter() {
            let sampled_embedding_space = sampled_embedder_state.embedding_spaces.get(type_id).unwrap();
            let sampled_value_field = value_field.sample(sampled_embedding_space);
            sampled_value_fields.insert(*type_id, sampled_value_field);
        }
        SampledValueFieldState {
            sampled_value_fields
        }
    }
    pub fn update_from_sampled(&mut self, mut sampled_value_field_state : SampledValueFieldState) {
        for (type_id, value_field) in self.value_fields.iter_mut() {
            let sampled_value_field = sampled_value_field_state.sampled_value_fields.remove(type_id).unwrap();
            value_field.update_from_sampled(sampled_value_field);
        }
    }

    //Deals in full vectors
    pub fn get_value_for_full_vector(&self, typed_vector : &TypedVector) -> f32 {
        let type_id = typed_vector.type_id;
        let value_field = self.get_value_field(type_id);
        let result = value_field.get_value_for_full_vector(&typed_vector.vec);
        result
    }

    pub fn get_value_field(&self, type_id : TypeId) -> &ValueField {
        self.value_fields.get(&type_id).unwrap()
    }
    pub fn get_value_field_mut(&mut self, type_id : TypeId) -> &mut ValueField {
        self.value_fields.get_mut(&type_id).unwrap()
    }
}
