use kinode_process_lib::http;
use kinode_process_lib::kernel_types::MessageType;
use kinode_process_lib::{
    await_message, call_init, get_blob, println, Address, LazyLoadBlob, Message, Request, Response,
};

mod ml_types;
use ml_types::{MlRequest, MlResponse};

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

fn ping_provider(provider_address: &Address) -> anyhow::Result<bool> {
    let response = Request::to(provider_address)
        .body(serde_json::to_vec(&MlRequest::Ping)?)
        .send_and_await_response(1)??;
    let MlResponse::Ping(is_provider) = serde_json::from_slice(response.body())? else {
        return Err(anyhow::anyhow!("expected Ping as Response to Ping, got {:?}", response));
    };
    Ok(is_provider)
}

fn handle_ws_message(
    connection: &mut Option<Connection>,
    message_body: &[u8],
) -> anyhow::Result<()> {
    match serde_json::from_slice::<http::HttpServerRequest>(message_body)? {
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
                        .body(serde_json::to_vec(&MlResponse::Run)?)
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

    if message.is_request() {
        match serde_json::from_slice(message.body()) {
            Ok(MlRequest::Run) => {
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
                    let provider_address = Address::new(provider, our.process.clone());
                    if ping_provider(&provider_address).is_ok_and(|is_live| is_live) {
                        Request::to(&provider_address)
                            .body(body)
                            .blob(blob)
                            .expects_response(15)
                            .send()?;

                        return Ok(());
                    }
                }
            }
            Ok(MlRequest::Ping) => {
                Response::new()
                    .body(serde_json::to_vec(&MlResponse::Ping(connection.is_some()))?)
                    .inherit(true)
                    .send()?;
                return Ok(());
            }
            Err(_) => {}
        }
    } else {
        match serde_json::from_slice(message.body()) {
            Ok(MlResponse::Run) => {
                Response::new()
                    .body(serde_json::to_vec(&MlResponse::Run)?)
                    .inherit(true)
                    .send()?;
                return Ok(());
            }
            Ok(MlResponse::Ping(_)) => {
                return Err(anyhow::anyhow!("unexpected Ping Response"));
            }
            Err(_) => {}
        }
    }

    handle_ws_message(connection, message.body())?;

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
