# zig-ai

A simple OpenAI API client for Zig with streaming support.

Find examples in the [`examples`](./examples) directory.

## Usage

With streaming:

```zig
const std = @import("std");
const OpenAI = @import("zig-ai");

pub fn main() !void {
    // ...

    var messages = std.ArrayList(OpenAI.Message).init(allocator);
    try messages.append(.{
        .role = "system",
        .content = "You are a helpful assistant",
    });
    try messages.append(.{
        .role = "user",
        .content = "User message here",
    });

    const payload = OpenAI.ChatPayload{
        .model = "gpt-4o",
        .messages = messages.items,
        .max_tokens = 1000,
        .temperature = 0.2,
    };

    var stream = try openai.streamChat(payload, false);
    defer stream.deinit();

    while (try stream.next()) |response| {
        // Stream the response to stdout
        if (response.choices[0].delta.content) |content| {
            try writer.writeAll(content);
            try buf_writer.flush();
        }
    }
}
```

## Installation

```bash
$ zig fetch --save git+https://github.com/FOLLGAD/zig-ai
```

and add `zig-ai` to your `build.zig` file:

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "stream-cli",
        .root_source_file = b.path("examples/stream-cli.zig"),
        .target = target,
        .optimize = optimize,
    });

    const module = b.dependency("zig-ai", .{
        .root_source_file = b.path("src/llm.zig"),
    });
    exe.root_module.addImport("zig-ai", module.module("zig-ai"));
}
```

## API Reference

### Chat Completions

```zig
// Non-streaming
const result = try openai.chat(payload, false);
defer result.deinit();

// Streaming
var stream = try openai.streamChat(payload, false);
defer stream.deinit();
while (try stream.next()) |response| { ... }
```

### Embeddings

```zig
const result = try openai.createEmbedding(.{
    .model = "text-embedding-3-small",
    .input = "The quick brown fox",
});
defer result.deinit();
const vector = result.value.data[0].embedding;
```

### Models

```zig
// List all models
const models = try openai.listModels();
defer models.deinit();

// Retrieve a specific model
const model = try openai.retrieveModel("gpt-4o");
defer model.deinit();
```

### Moderations

```zig
const result = try openai.createModeration(.{
    .input = "Some text to check",
});
defer result.deinit();
const flagged = result.value.results[0].flagged;
```

### Image Generation

```zig
const result = try openai.createImage(.{
    .prompt = "A white siamese cat",
    .model = "dall-e-3",
    .n = 1,
    .size = "1024x1024",
});
defer result.deinit();
const url = result.value.data[0].url;
```

See the `examples` directory for more usage examples.
