use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct Multiple<T> {
    pub elem : T,
    pub count : usize
}

impl<T : Clone> Clone for Multiple<T> {
    fn clone(&self) -> Self {
        Multiple {
            elem : self.elem.clone(),
            count : self.count
        }
    }
}
