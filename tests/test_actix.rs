use actix::prelude::*;
use letsearch::actors::model_actor::ModelManagerActor;
use tokio::runtime::Runtime;

fn main() {
    let rt = Runtime::new().unwrap();
    rt.block_on(async {
        // try to spawn an actor
        let addr = ModelManagerActor::new().start();
        println!("Actor started");
    });
}
