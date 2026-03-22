import requests
import json
import base64
from typing import Dict, Any

# ------------------------------------------------------------------
# Agent 1: Medical Image Analysis Agent (Client)
# This represents our fine-tuned LLaMA-Factory model exposed via an API.
# ------------------------------------------------------------------
class MedicalImageAnalysisAgent:
    def __init__(self, api_base_url: str):
        """
        Initializes the agent with the API endpoint of the fine-tuned model.
        """
        self.api_url = f"{api_base_url}/v1/chat/completions"

    def analyze_image(self, image_path: str, question: str) -> Dict[str, Any]:
        """
        Sends an image and a question to the model API and gets a response.
        """
        print(f"\n[Medical Image Analysis Agent] Received task: Analyze '{image_path}' with question: '{question}'")
        
        # Encode image to base64
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        except FileNotFoundError:
            return {"error": f"Image file not found at {image_path}"}

        # Prepare the request payload for the LLaMA-Factory API
        payload = {
            "model": "qwen2.5-vl-7b-instruct", # This should match the model served by the API
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}
                    ]
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.1,
            "top_p": 0.8
        }

        headers = {"Content-Type": "application/json"}
        
        print("[Medical Image Analysis Agent] Sending request to model API...")
        try:
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status() # Raise an exception for bad status codes
            
            api_response = response.json()
            analysis_content = api_response['choices'][0]['message']['content']
            
            print("[Medical Image Analysis Agent] Received analysis from model.")
            # In a real system, you would parse this content into a structured format (e.g., JSON)
            # For this example, we'll just return the raw text.
            return {"analysis": analysis_content}

        except requests.exceptions.RequestException as e:
            error_message = f"API request failed: {e}"
            print(f"[Medical Image Analysis Agent] {error_message}")
            return {"error": error_message}
        except (KeyError, IndexError) as e:
            error_message = f"Failed to parse API response: {e}. Response: {response.text}"
            print(f"[Medical Image Analysis Agent] {error_message}")
            return {"error": error_message}


# ------------------------------------------------------------------
# Agent 2: Coordinator Agent
# This agent orchestrates the workflow.
# ------------------------------------------------------------------
class CoordinatorAgent:
    def __init__(self, image_agent: MedicalImageAnalysisAgent):
        """
        Initializes the coordinator with the agents it can delegate tasks to.
        """
        self.image_agent = image_agent
        # In a real system, you would also initialize the KnowledgeBaseAgent and ReportGeneratorAgent here.

    def run_workflow(self, user_image_path: str, user_question: str):
        """
        Executes the multi-agent workflow for a user request.
        """
        print("\n--- Starting Multi-Agent Workflow ---")
        print(f"[Coordinator Agent] Received user request for image '{user_image_path}'")

        # Step 1: Delegate image analysis task
        print("[Coordinator Agent] Delegating to Medical Image Analysis Agent...")
        analysis_result = self.image_agent.analyze_image(user_image_path, user_question)

        if "error" in analysis_result:
            print(f"[Coordinator Agent] Workflow failed. Reason: {analysis_result['error']}")
            return

        # Step 2: (Simulated) Delegate to Knowledge Base Agent
        # In a real system, you would parse `analysis_result` to extract keywords
        # and then query the knowledge base agent.
        print("\n[Coordinator Agent] Task 'Analyze Image' completed. Result received.")
        print("[Coordinator Agent] (Simulated) Delegating to Medical Knowledge Base Agent...")
        knowledge_base_info = self.query_knowledge_base(analysis_result['analysis'])
        print("[Coordinator Agent] (Simulated) Received info from Knowledge Base.")

        # Step 3: (Simulated) Delegate to Report Generation Agent
        print("\n[Coordinator Agent] (Simulated) Delegating to Report Generation Agent to synthesize final report...")
        final_report = self.generate_report(analysis_result['analysis'], knowledge_base_info)
        
        print("\n--- Workflow Finished ---")
        print("\nFinal Report:")
        print("---------------------------------")
        print(final_report)
        print("---------------------------------")

    def query_knowledge_base(self, analysis: str) -> Dict[str, str]:
        # This is a mock function. A real implementation would query a database or another API.
        if "tumor" in analysis.lower() or "lesion" in analysis.lower():
            return {
                "differential_diagnosis": "Gastric Cancer, GIST, Lymphoma.",
                "recommendation": "Endoscopic ultrasound (EUS) with fine-needle aspiration (FNA) is recommended for definitive diagnosis."
            }
        return {"info": "No specific knowledge base entries found for the analysis."}

    def generate_report(self, analysis: str, kb_info: Dict[str, str]) -> str:
        # This is a mock function. A real implementation would use an LLM to generate a coherent report.
        report = f"**1. Image Analysis Findings:**\n{analysis}\n\n"
        report += "**2. Knowledge Base Information:**\n"
        for key, value in kb_info.items():
            report += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        return report

# ------------------------------------------------------------------
# Main execution block
# ------------------------------------------------------------------
if __name__ == "__main__":
    # IMPORTANT: Before running this, you must start the LLaMA-Factory API server using:
    # llamafactory-cli api --model_name_or_path qwen/Qwen2.5-7B-VL-Instruct --adapter_path saves/qwen2_5vl/lora/sft-20240801-120000
    
    # Configuration
    LLAMA_FACTORY_BASE_URL = "http://localhost:8000" # Default LLaMA-Factory API address
    
    # This is a placeholder path. Replace it with an actual path to a medical image.
    # For example: "D:\\weng\\PubMedVision\\images_0\\images\\PMC1000000_PMC7033654_gkz1101-fig1.jpg"
    # To make this script runnable, let's create a dummy image file.
    DUMMY_IMAGE_PATH = "dummy_medical_image.jpg"
    try:
        with open(DUMMY_IMAGE_PATH, "wb") as f:
            # A simple 1x1 black pixel JPEG image
            f.write(base64.b64decode('/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAIBAQIBAQICAgICAgICAwUDAwMDAwYEBAMFBwYHBwcGBwcICQsJCAgKCAcHCg0KCgsMDAwMBwkODw0MDgsMDAz/2wBDAQICAgMDAwYDAwYMCAcIDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAz/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaamqKpqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD86qKKKAP/2Q=='))
    except Exception as e:
        print(f"Could not create dummy image file: {e}")


    USER_QUESTION = "What abnormalities do you see in this image?"

    # 1. Initialize the agents
    image_agent = MedicalImageAnalysisAgent(api_base_url=LLAMA_FACTORY_BASE_URL)
    coordinator = CoordinatorAgent(image_agent=image_agent)

    # 2. Run the workflow
    # NOTE: This will fail if the API server is not running.
    coordinator.run_workflow(user_image_path=DUMMY_IMAGE_PATH, user_question=USER_QUESTION)