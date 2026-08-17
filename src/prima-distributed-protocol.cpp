#include "prima-distributed-protocol.h"

#include <limits>

namespace prima {
namespace {

constexpr size_t frame_prefix_bytes = sizeof(uint64_t);
constexpr size_t body_header_bytes = 4 + 2 + 2 + 4 + 4 + 8 + 2 + 2 + 8;

void append_u16(std::vector<uint8_t> & out, uint16_t value) {
    out.push_back(static_cast<uint8_t>(value));
    out.push_back(static_cast<uint8_t>(value >> 8));
}

void append_u32(std::vector<uint8_t> & out, uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<uint8_t>(value >> (8 * i)));
    }
}

void append_u64(std::vector<uint8_t> & out, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<uint8_t>(value >> (8 * i)));
    }
}

bool read_u16(const uint8_t * data, size_t size, size_t & offset, uint16_t & value, std::string & error) {
    if (offset > size || size - offset < 2) {
        error = "truncated u16";
        return false;
    }
    value = static_cast<uint16_t>(data[offset]) |
            static_cast<uint16_t>(data[offset + 1]) << 8;
    offset += 2;
    return true;
}

bool read_u32(const uint8_t * data, size_t size, size_t & offset, uint32_t & value, std::string & error) {
    if (offset > size || size - offset < 4) {
        error = "truncated u32";
        return false;
    }
    value = 0;
    for (int i = 0; i < 4; ++i) {
        value |= static_cast<uint32_t>(data[offset + i]) << (8 * i);
    }
    offset += 4;
    return true;
}

bool read_u64(const uint8_t * data, size_t size, size_t & offset, uint64_t & value, std::string & error) {
    if (offset > size || size - offset < 8) {
        error = "truncated u64";
        return false;
    }
    value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(data[offset + i]) << (8 * i);
    }
    offset += 8;
    return true;
}

bool valid_message_type(uint16_t value) {
    return value == static_cast<uint16_t>(message_type::hello) ||
           value == static_cast<uint16_t>(message_type::batch_meta) ||
           value == static_cast<uint16_t>(message_type::hidden_state) ||
           value == static_cast<uint16_t>(message_type::token) ||
           value == static_cast<uint16_t>(message_type::reset) ||
           value == static_cast<uint16_t>(message_type::error);
}

bool valid_data_type(uint16_t value) {
    return value <= static_cast<uint16_t>(data_type::i32);
}

} // namespace

std::vector<uint8_t> encode_frame(const message & message) {
    if (message.payload.size() > max_payload_bytes || message.shape.size() > std::numeric_limits<uint16_t>::max()) {
        return {};
    }

    const uint64_t dims_bytes = static_cast<uint64_t>(message.shape.size()) * sizeof(uint64_t);
    const uint64_t body_size = body_header_bytes + dims_bytes + message.payload.size();
    if (body_size > static_cast<uint64_t>(frame_prefix_bytes) + max_payload_bytes ||
        body_size > std::numeric_limits<size_t>::max() - frame_prefix_bytes) {
        return {};
    }

    std::vector<uint8_t> frame;
    frame.reserve(frame_prefix_bytes + static_cast<size_t>(body_size));
    append_u64(frame, body_size);
    append_u32(frame, protocol_magic);
    append_u16(frame, protocol_version);
    append_u16(frame, static_cast<uint16_t>(message.type));
    append_u32(frame, message.flags);
    append_u32(frame, message.rank);
    append_u64(frame, message.sequence);
    append_u16(frame, static_cast<uint16_t>(message.dtype));
    append_u16(frame, static_cast<uint16_t>(message.shape.size()));
    append_u64(frame, message.payload.size());
    for (uint64_t dimension : message.shape) {
        append_u64(frame, dimension);
    }
    frame.insert(frame.end(), message.payload.begin(), message.payload.end());
    return frame;
}

bool decode_frame(const uint8_t * data, size_t size, message & output, std::string & error) {
    error.clear();
    if (data == nullptr || size < frame_prefix_bytes) {
        error = "truncated frame prefix";
        return false;
    }

    size_t offset = 0;
    uint64_t body_size = 0;
    if (!read_u64(data, size, offset, body_size, error)) {
        return false;
    }
    if (body_size > frame_prefix_bytes + max_payload_bytes || body_size > size - frame_prefix_bytes) {
        error = "truncated frame body";
        return false;
    }
    if (body_size < body_header_bytes) {
        error = "truncated frame header";
        return false;
    }

    const size_t body_end = frame_prefix_bytes + static_cast<size_t>(body_size);
    if (body_end > size) {
        error = "truncated frame body";
        return false;
    }

    uint32_t magic = 0;
    uint16_t version = 0;
    uint16_t type = 0;
    uint16_t dtype = 0;
    uint16_t dimensions = 0;
    uint64_t payload_size = 0;
    if (!read_u32(data, body_end, offset, magic, error) ||
        !read_u16(data, body_end, offset, version, error) ||
        !read_u16(data, body_end, offset, type, error) ||
        !read_u32(data, body_end, offset, output.flags, error) ||
        !read_u32(data, body_end, offset, output.rank, error) ||
        !read_u64(data, body_end, offset, output.sequence, error) ||
        !read_u16(data, body_end, offset, dtype, error) ||
        !read_u16(data, body_end, offset, dimensions, error) ||
        !read_u64(data, body_end, offset, payload_size, error)) {
        return false;
    }

    if (magic != protocol_magic) {
        error = "invalid protocol magic";
        return false;
    }
    if (version != protocol_version) {
        error = "unsupported protocol version";
        return false;
    }
    if (!valid_message_type(type)) {
        error = "invalid message type";
        return false;
    }
    if (!valid_data_type(dtype)) {
        error = "invalid data type";
        return false;
    }
    if (payload_size > max_payload_bytes) {
        error = "payload exceeds limit";
        return false;
    }

    const uint64_t required = static_cast<uint64_t>(offset) +
                              static_cast<uint64_t>(dimensions) * sizeof(uint64_t) +
                              payload_size;
    if (required > body_end) {
        error = "truncated frame contents";
        return false;
    }

    output.type = static_cast<message_type>(type);
    output.dtype = static_cast<data_type>(dtype);
    output.shape.clear();
    output.shape.reserve(dimensions);
    for (uint16_t i = 0; i < dimensions; ++i) {
        uint64_t dimension = 0;
        if (!read_u64(data, body_end, offset, dimension, error)) {
            return false;
        }
        output.shape.push_back(dimension);
    }

    output.payload.assign(data + offset, data + offset + static_cast<size_t>(payload_size));
    return true;
}

} // namespace prima
