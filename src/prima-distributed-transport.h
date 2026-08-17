#pragma once

#include "prima-distributed-protocol.h"

#include <cstdint>
#include <string>

namespace prima {

class tcp_channel {
public:
    tcp_channel() = default;
    explicit tcp_channel(int fd);
    ~tcp_channel();

    tcp_channel(const tcp_channel &) = delete;
    tcp_channel & operator=(const tcp_channel &) = delete;
    tcp_channel(tcp_channel && other) noexcept;
    tcp_channel & operator=(tcp_channel && other) noexcept;

    bool valid() const;
    bool send(const message & message, std::string & error);
    bool receive(message & message, std::string & error);

    static tcp_channel connect_host(const std::string & host, uint16_t port, std::string & error);
    static tcp_channel connect_loopback(uint16_t port, std::string & error);

private:
    int fd_ = -1;

    bool write_all(const uint8_t * data, size_t size, std::string & error);
    bool read_all(uint8_t * data, size_t size, std::string & error);
};

class tcp_listener {
public:
    tcp_listener() = default;
    ~tcp_listener();

    tcp_listener(const tcp_listener &) = delete;
    tcp_listener & operator=(const tcp_listener &) = delete;
    tcp_listener(tcp_listener && other) noexcept;
    tcp_listener & operator=(tcp_listener && other) noexcept;

    static tcp_listener bind_host(const std::string & host, uint16_t port);
    static tcp_listener bind_loopback(uint16_t port);
    bool valid() const;
    uint16_t port() const;
    tcp_channel accept(std::string & error);

private:
    int fd_ = -1;
    uint16_t port_ = 0;

    tcp_listener(int fd, uint16_t port);
};

} // namespace prima
