import asyncio
from fastmcp import Client
from mcp_server import mcp

async def test():
    async with Client(mcp) as client:
        print("=== 1. Active Tools ===")
        tools = await client.list_tools()
        for t in tools:
            print(f" - {t.name}")

        print("\n=== 2. Empirical Benchmark (yolov8n) ===")
        perf = await client.call_tool("get_model_performance", {"model": "yolov8n"})
        print(perf)

        print("\n=== 3. Hardware Recommendation ===")
        rec = await client.call_tool("recommend_deployment", {
            "target_hardware": "raspberry_pi_or_cpu_edge",
            "annotated_training_images_available": 2000,
            "priority": "fastest_edge_inference"
        })
        print(rec)

        print("\n=== 4. Runtime Inference (detect_mortality) ===")
        det = await client.call_tool("detect_mortality", {
            "image_path": "test_images/tl_0031_0153_20221022_001226_jpg.rf.5a6ee26f647296aab6267d96b17ced6f.jpg",
            "model": "yolov8n",
            "weight_format": "onnx",
            "conf": 0.25
        })
        print(det)

if __name__ == "__main__":
    asyncio.run(test())
