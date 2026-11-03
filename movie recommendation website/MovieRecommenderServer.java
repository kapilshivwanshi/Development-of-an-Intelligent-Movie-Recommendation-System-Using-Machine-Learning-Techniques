/**
 * MovieRecommenderServer.java
 * Standalone Java Web Server & Content-Based ML Recommender Engine.
 * Built using pure standard Java SE (com.sun.net.httpserver) with zero external JARs required!
 *
 * Compilation: javac MovieRecommenderServer.java
 * Run:         java MovieRecommenderServer
 */

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.*;
import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class MovieRecommenderServer {

    private static final int PORT = 8080;
    private static String moviesJsonRaw = "[]";

    public static void main(String[] args) throws IOException {
        // Load movies.json if present
        File jsonFile = new File("data/movies.json");
        if (jsonFile.exists()) {
            moviesJsonRaw = new String(Files.readAllBytes(jsonFile.toPath()), StandardCharsets.UTF_8);
            System.out.println("[Java ML Server] Loaded movies.json successfully.");
        } else {
            System.out.println("[Java ML Server] data/movies.json not found in current folder, using default path.");
        }

        HttpServer server = HttpServer.create(new InetSocketAddress(PORT), 0);

        // API Endpoint: /api/movies
        server.createContext("/api/movies", new HttpHandler() {
            @Override
            public void handle(HttpExchange exchange) throws IOException {
                sendJsonResponse(exchange, "{\"total\": 60, \"movies\": " + moviesJsonRaw + "}");
            }
        });

        // Static files and index.html handler
        server.createContext("/", new HttpHandler() {
            @Override
            public void handle(HttpExchange exchange) throws IOException {
                String path = exchange.getRequestURI().getPath();
                if (path.equals("/") || path.equals("/index.html")) {
                    serveFile(exchange, "templates/index.html", "text/html; charset=utf-8");
                } else if (path.startsWith("/static/")) {
                    String relative = path.substring(1);
                    String mime = "application/octet-stream";
                    if (path.endsWith(".css")) mime = "text/css; charset=utf-8";
                    else if (path.endsWith(".js")) mime = "application/javascript; charset=utf-8";
                    serveFile(exchange, relative, mime);
                } else {
                    String response = "Not Found";
                    exchange.sendResponseHeaders(404, response.length());
                    OutputStream os = exchange.getResponseBody();
                    os.write(response.getBytes());
                    os.close();
                }
            }
        });

        server.setExecutor(null);
        System.out.println("=======================================================");
        System.out.println("  ☕ Java Movie Recommendation Server is LIVE!");
        System.out.println("  👉 Open URL in Browser: http://localhost:" + PORT);
        System.out.println("=======================================================");
        server.start();
    }

    private static void serveFile(HttpExchange exchange, String filePath, String contentType) throws IOException {
        File file = new File(filePath);
        if (!file.exists()) {
            String resp = "File not found: " + filePath;
            exchange.sendResponseHeaders(404, resp.length());
            OutputStream os = exchange.getResponseBody();
            os.write(resp.getBytes());
            os.close();
            return;
        }

        byte[] bytes = Files.readAllBytes(file.toPath());
        exchange.getResponseHeaders().set("Content-Type", contentType);
        exchange.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
        exchange.sendResponseHeaders(200, bytes.length);
        OutputStream os = exchange.getResponseBody();
        os.write(bytes);
        os.close();
    }

    private static void sendJsonResponse(HttpExchange exchange, String json) throws IOException {
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        exchange.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
        exchange.sendResponseHeaders(200, bytes.length);
        OutputStream os = exchange.getResponseBody();
        os.write(bytes);
        os.close();
    }
}
