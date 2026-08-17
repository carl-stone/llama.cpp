#include "../src/prima-distributed-protocol.h"
#include "../src/prima-distributed-transport.h"

#include <cassert>
#include <cstdint>
#include <thread>

int main() {
    prima::tcp_listener listener = prima::tcp_listener::bind_loopback(0);
    assert(listener.valid());
    const uint16_t port = listener.port();
    assert(port != 0);

    prima::message sent;
    sent.type = prima::message_type::hidden_state;
    sent.rank = 0;
    sent.sequence = 42;
    sent.dtype = prima::data_type::f32;
    sent.shape = {1, 1, 16};
    sent.payload.assign(64, 0xab);

    std::thread server([&] {
        std::string error;
        prima::tcp_channel peer = listener.accept(error);
        assert(peer.valid());

        prima::message received;
        assert(peer.receive(received, error));
        assert(received.type == sent.type);
        assert(received.sequence == sent.sequence);
        assert(received.shape == sent.shape);
        assert(received.payload == sent.payload);
        assert(peer.send(sent, error));
    });

    std::string error;
    prima::tcp_channel client = prima::tcp_channel::connect_loopback(port, error);
    assert(client.valid());
    assert(client.send(sent, error));

    prima::message echoed;
    assert(client.receive(echoed, error));
    assert(echoed.sequence == sent.sequence);
    assert(echoed.payload == sent.payload);

    server.join();
    return 0;
}
