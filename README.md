# Redis-Compatible Cache Server

A custom Redis-compatible cache server implementation written in C++ for learning systems programming and network protocols.

## Project Overview

This project implements a Redis-like in-memory key-value store from scratch using low-level C++ socket programming. It's designed as a learning exercise to understand:

- TCP socket programming with IPv4/IPv6 dual-stack support
- Network protocol design and implementation
- Memory management and data structures
- Concurrent client handling
- Redis protocol compatibility

## Features

### Current Implementation
-  **Dual-stack networking**: Supports both IPv4 and IPv6 connections
-  **TCP socket server**: Robust connection handling with proper error checking
-  **Socket options**: SO_REUSEADDR for development convenience
-  **Client connection management**: Accept, process, and close connections gracefully

### Planned Features
-  **Redis Protocol (RESP)**: Compatible with Redis clients
-  **Core Commands**: GET, SET, DEL, EXISTS, EXPIRE
-  **Data Types**: Strings, Lists, Sets, Hashes
-  **Persistence**: RDB-style snapshots
-  **Concurrency**: Multi-threaded client handling
-  **Memory Management**: LRU eviction policies

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Redis Client  │────│  Network Layer  │────│  Cache Engine   │
│                 │    │                 │    │                 │
│ - redis-cli     │    │ - IPv4/IPv6     │    │ - Key-Value     │
│ - Applications  │    │ - TCP Sockets   │    │ - Commands      │
│ - Libraries     │    │ - RESP Protocol │    │ - Data Types    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Technical Details

### Network Layer
- **IPv4 Endpoint**: `0.0.0.0:1234` (all interfaces)
- **IPv6 Endpoint**: `[::]:1235` (all interfaces) 
- **Protocol**: TCP with proper connection lifecycle management
- **Endianness**: Network byte order conversion using `htons()`/`htonl()`

### Socket Programming Features
```cpp
// Dual-stack socket creation
int fd4 = socket(AF_INET, SOCK_STREAM, 0);    // IPv4
int fd6 = socket(AF_INET6, SOCK_STREAM, 0);   // IPv6

// Address reuse for development
setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &val, sizeof(val));

// Proper error handling
if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
    die("bind failed");
}
```

## Prerequisites

- **Compiler**: GCC 7+ or Clang 6+ with C++11 support
- **OS**: Linux, macOS, or WSL2 (POSIX socket support required)
- **Dependencies**: Standard C++ library only

## Building and Running

### Compile
```bash
g++ -std=c++11 -Wall -Wextra -o redis-server main.cpp
```

### Run Server
```bash
./redis-server
```

### Test Connection
```bash
# IPv4 connection
telnet localhost 1234

# IPv6 connection (if supported)
telnet ::1 1235

# Using netcat
nc localhost 1234
```

## Testing

### Basic Connection Test
```bash
# Terminal 1: Start server
./redis-server

# Terminal 2: Connect and test
echo "Hello Redis!" | nc localhost 1234
```

### Multiple Connections
The server handles one connection at a time currently. Each connection is:
1. Accepted on the listening socket
2. Processed by `do_something()` function
3. Gracefully closed

## Project Structure

```
redis-cache-server/
├── README.md           # This file
├── main.cpp           # Main server implementation
├── Makefile           # Build configuration (planned)
├── src/               # Source files (planned)
│   ├── server.cpp     # Server logic
│   ├── protocol.cpp   # Redis protocol parser
│   └── cache.cpp      # Cache implementation
├── include/           # Header files (planned)
└── tests/             # Unit tests (planned)
```

## Learning Objectives

This project covers key systems programming concepts:

### Socket Programming
- Creating and configuring TCP sockets
- Binding to network interfaces
- Listening for incoming connections
- Accepting and handling client connections
- Proper resource cleanup and error handling

### Network Protocols
- Understanding IPv4 vs IPv6 addressing
- Network byte order vs host byte order
- Socket address structures (`sockaddr_in`, `sockaddr_in6`)
- TCP connection lifecycle

### System Programming
- File descriptor management
- Error handling with `perror()` and proper cleanup
- Signal handling (planned)
- Memory management for concurrent access

## Development Roadmap

### Phase 1: Foundation 
- [x] Basic TCP server with IPv4/IPv6 support
- [x] Connection accept/close cycle
- [x] Error handling and logging

### Phase 2: Protocol Implementation
- [ ] RESP (Redis Serialization Protocol) parser
- [ ] Basic command structure (GET, SET, PING)
- [ ] Client-server communication

### Phase 3: Core Functionality 
- [ ] In-memory key-value storage
- [ ] String operations
- [ ] TTL (Time To Live) support

### Phase 4: Advanced Features 
- [ ] Multi-threading for concurrent clients
- [ ] Additional data types (Lists, Sets, Hashes)
- [ ] Persistence mechanisms
- [ ] Configuration file support

## Contributing

This is primarily a learning project, but contributions and suggestions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Resources

- [Redis Protocol Specification](https://redis.io/docs/reference/protocol-spec/)
- [Beej's Guide to Network Programming](https://beej.us/guide/bgnet/)
- [TCP/IP Illustrated](https://www.amazon.com/TCP-Illustrated-Volume-Implementation/dp/0201633469)
- [The Linux Programming Interface](https://man7.org/tlpi/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by the original Redis implementation by Salvatore Sanfilippo
- Socket programming examples from various online tutorials
- C++ systems programming best practices from the community

---

**Note**: This is an educational project and is not intended for production use. For production Redis needs, please use the official Redis server.
