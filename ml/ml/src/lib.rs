use kinode_process_lib::http;
use kinode_process_lib::kernel_types::MessageType;
use kinode_process_lib::{
    await_message, call_init, get_blob, println, Address, LazyLoadBlob, Message, Request, Response,
};

mod ml_types;

wit_bindgen::generate!({
    path: "wit",
    world: "process",
    exports: {
        world: Component,
    },
});

struct Connection {
    channel_id: u32,
}

fn is_expected_channel_id(
    connection: &Option<Connection>,
    channel_id: &u32,
) -> anyhow::Result<bool> {
    let Some(Connection { channel_id: ref current_channel_id }) = connection else {
        return Err(anyhow::anyhow!("a"));
    };

    Ok(channel_id == current_channel_id)
}

fn get_providers() -> anyhow::Result<Vec<String>> {
    // TODO: make this real
    Ok(vec!["fake2.os".into()])
}

fn ping_provider(provider: &str, our: &Address) -> anyhow::Result<bool> {
    // TODO: make this real
    Ok(true)
}

fn handle_ws_message(
    connection: &mut Option<Connection>,
    message: Message,
) -> anyhow::Result<()> {
    match serde_json::from_slice::<http::HttpServerRequest>(message.body())? {
        http::HttpServerRequest::Http(_) => {
            // TODO: response?
            return Err(anyhow::anyhow!("b"));
        }
        http::HttpServerRequest::WebSocketOpen { channel_id, .. } => {
            *connection = Some(Connection { channel_id });
        }
        http::HttpServerRequest::WebSocketClose(ref channel_id) => {
            if !is_expected_channel_id(connection, channel_id)? {
                // TODO: response?
                return Err(anyhow::anyhow!("c"));
            }
            *connection = None;
        }
        http::HttpServerRequest::WebSocketPush { ref channel_id, ref message_type } => {
            if !is_expected_channel_id(connection, channel_id)? {
                // TODO: response?
                return Err(anyhow::anyhow!("d"));
            }
            match message_type {
                http::WsMessageType::Binary => {
                    Response::new()
                        .body(serde_json::to_vec(&serde_json::json!("Run"))?)
                        .inherit(true)
                        .send()?;
                }
                http::WsMessageType::Close => {
                    *connection = None;
                }
                e => {
                    // TODO: response; handle other types?
                    return Err(anyhow::anyhow!("e: {:?}", e));
                }
            }
        }
    }
    Ok(())
}

fn handle_message(
    our: &Address,
    connection: &mut Option<Connection>,
) -> anyhow::Result<()> {
    let message = await_message()?;

    if serde_json::json!("Run") != serde_json::from_slice::<serde_json::Value>(message.body())? {
        handle_ws_message(connection, message)?;
    } else {
        if message.is_request() {
            if let Some(Connection { channel_id }) = connection {
                Request::to("our@http_server:distro:sys".parse::<Address>()?)
                    .body(serde_json::to_vec(&http::HttpServerAction::WebSocketExtPushOutgoing {
                        channel_id: *channel_id,
                        message_type: http::WsMessageType::Binary,
                        desired_reply_type: MessageType::Response,
                    })?)
                    .expects_response(15)
                    .inherit(true)
                    .send()?;
                return Ok(());
            };
            // Read in blob since pinging providers will clear it (new messages overwrite).
            let Some(blob) = get_blob() else {
                // No blob -> no message.
                return Ok(());
            };
            let body = message.body();

            let providers = get_providers()?;
            for provider in providers {
                if ping_provider(&provider, our).is_ok_and(|is_live| is_live) {
                    let provider_address = Address::new(provider, our.process.clone());
                    Request::to(provider_address)
                        .body(body)
                        .blob(blob)
                        .expects_response(15)
                        .send()?;

                    return Ok(());
                }
            }
        } else {
            Response::new()
                .body(serde_json::to_vec(&serde_json::json!("Run"))?)
                .inherit(true)
                .send()?;
        }
    }

    Ok(())
}

call_init!(init);
fn init(our: Address) {
    println!("{our}: begin");

    let mut connection: Option<Connection> = None;

    http::bind_ext_path("/").unwrap();

    loop {
        match handle_message(&our, &mut connection) {
            Ok(()) => {}
            Err(e) => {
                println!("{our}: error: {:?}", e);
            }
        };
    }
}
