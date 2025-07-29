#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <arpa/inet.h>
#include <cstdlib>
using namespace std;

void die(const char* msg) {
    perror(msg);
    exit(1);
}
//Can replace read and write with send/recv for more control over TCP
//For example, send/recv can handle flags like MSG_NOSIGNAL to prevent SIGPIPE
static void do_something(int fd) {
    // Placeholder for processing client connection
    // In a real application, you would read/write data here
    char[64] rbuf = {};
    ssize_t n = read(fd, rbuf, sizeof(rbuf) - 1);
    cout << "Processing client on fd: " << fd << endl;
}

int main() {
    // Socket option value for SO_REUSEADDR - allows reusing the address
    int val = 1;
    
    // ===== IPv4 SOCKET SETUP =====
    cout << "Setting up IPv4 socket..." << endl;
    
    // Create IPv4 TCP socket
    int fd4 = socket(AF_INET, SOCK_STREAM, 0);
    if (fd4 < 0) die("socket IPv4");
    
    // Allow socket to reuse address (prevents "Address already in use" errors)
    setsockopt(fd4, SOL_SOCKET, SO_REUSEADDR, &val, sizeof(val));
    
    // IPv4 address structure
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;                    // AF_INET for IPv4
    addr.sin_port = htons(1234);                  // port in big-endian (network byte order)
    addr.sin_addr.s_addr = htonl(INADDR_ANY);     // wildcard IP 0.0.0.0 (listen on all interfaces)
    
    // Bind socket to address - cast sockaddr_in* to sockaddr* for generic bind() call
    if (bind(fd4, (const struct sockaddr *)&addr, sizeof(addr)) < 0) {
        die("bind IPv4");
    }
    
    // Start listening for connections (backlog of 5)
    if (listen(fd4, SOMAXCONN) < 0) die("listen IPv4");
    
    // ===== IPv6 SOCKET SETUP =====
    cout << "Setting up IPv6 socket..." << endl;
    
    // Create IPv6 TCP socket
    int fd6 = socket(AF_INET6, SOCK_STREAM, 0);
    if (fd6 < 0) die("socket IPv6");
    
    // Allow socket to reuse address
    setsockopt(fd6, SOL_SOCKET, SO_REUSEADDR, &val, sizeof(val));
    
    // IPv6 address structure
    struct sockaddr_in6 addr6 = {};
    addr6.sin6_family = AF_INET6;                 // AF_INET6 for IPv6
    addr6.sin6_port = htons(1235);                // port in big-endian (different port from IPv4)
    addr6.sin6_addr = in6addr_any;                // IPv6 wildcard address (::)
    addr6.sin6_flowinfo = 0;                      // flow info (usually ignored)
    addr6.sin6_scope_id = 0;                      // scope ID (usually ignored)
    
    // Bind IPv6 socket - same cast pattern as IPv4
    if (bind(fd6, (struct sockaddr*)&addr6, sizeof(addr6)) < 0) {
        die("bind IPv6");
    }
    
    // Start listening for IPv6 connections
    if (listen(fd6, 5) < 0) die("listen IPv6");
    
    cout << "Servers listening on:" << endl;
    cout << "  IPv4: 0.0.0.0:1234" << endl;
    cout << "  IPv6: [::]:1235" << endl;
    
    // ===== ENDIANNESS CONVERSION FUNCTIONS (for reference) =====
    /*
     * Host-to-Network and Network-to-Host conversion functions:
     * 
     * uint16_t htons(uint16_t hostshort);    // host to network short (16-bit)
     * uint32_t htonl(uint32_t hostlong);     // host to network long (32-bit)
     * uint16_t ntohs(uint16_t netshort);     // network to host short
     * uint32_t ntohl(uint32_t netlong);      // network to host long
     * 
     * Alternative big-endian functions:
     * uint32_t htobe32(uint32_t host_32bits);      // host to big-endian
     * uint32_t be32toh(uint32_t big_endian_32bits); // big-endian to host
     * 
     * Why do we need these?
     * - Network protocols use big-endian byte order (most significant byte first)
     * - Host machines might use little-endian (Intel x86) or big-endian
     * - These functions ensure correct byte order regardless of host architecture
     */
    
    // ===== SOCKET ADDRESS STRUCTURES (for reference) =====
    /*
     * IPv4 structure:
     * struct sockaddr_in {
     *     uint16_t sin_family;        // AF_INET
     *     uint16_t sin_port;          // port in big-endian
     *     struct in_addr sin_addr;    // IPv4 address
     * };
     * 
     * struct in_addr {
     *     uint32_t s_addr;            // IPv4 address in big-endian
     * };
     * 
     * IPv6 structure:
     * struct sockaddr_in6 {
     *     uint16_t sin6_family;       // AF_INET6
     *     uint16_t sin6_port;         // port in big-endian
     *     uint32_t sin6_flowinfo;     // flow information (usually 0)
     *     struct in6_addr sin6_addr;  // IPv6 address
     *     uint32_t sin6_scope_id;     // scope ID (usually 0)
     * };
     * 
     * struct in6_addr {
     *     uint8_t s6_addr[16];        // IPv6 address (128 bits)
     * };
     */
    
    // ===== MAIN SERVER LOOP =====
    /*
     * NOTE: This simplified example only accepts connections on the IPv4 socket.
     * In a real application, you would use select(), poll(), or epoll() to 
     * monitor both sockets simultaneously and accept connections from either.
     */
    cout << "Waiting for connections on IPv4 socket..." << endl;
    
    while(true) {
        // Accept incoming connection
        // accept() returns a new socket file descriptor for the client connection
        struct sockaddr_in client_addr = {};
        socklen_t addrlen = sizeof(client_addr);
        int conn_fd = accept(fd4, (struct sockaddr*)&client_addr, &addrlen);
        if (conn_fd < 0) {continue;} //Error   
        cout << "New connection accepted!" << endl;
        
        // TODO: do_something_with(conn_fd);
        // This is where you would:
        // - Read data from the client: recv() or read()
        // - Process the data
        // - Send response back: send() or write()
        
        // Close the client connection
        do_something(conn_fd);
        close(conn_fd);
        cout << "Connection closed." << endl;
    }
    
    // Cleanup (never reached in this example due to infinite loop)
    close(fd4);
    close(fd6);
    return 0;
}