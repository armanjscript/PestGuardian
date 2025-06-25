# PestGuardian: AI-Powered Pest Detection and Elimination Assistant

## Description

PestGuardian is a web-based application designed to assist farmers and gardeners in identifying pests in their crops and providing actionable recommendations for their elimination. By leveraging advanced computer vision and natural language processing, PestGuardian allows users to upload images of pests, detects the pests using a state-of-the-art YOLO model, and generates practical elimination methods using web search and AI-driven recommendations.

This project combines cutting-edge technologies to create a user-friendly tool that simplifies pest management, making it accessible even to those without extensive technical expertise.

## Technologies Used

| Technology | Purpose |
|------------|---------|
| **Streamlit** | Creates the interactive web interface for user interaction. |
| **Ultralytics YOLO** | Detects pests in uploaded images using the YOLO model. |
| **LangChain & LangChain-Ollama** | Powers natural language processing with the Qwen2.5 model for generating recommendations. |
| **GoogleSerperAPIWrapper** | Performs web searches to find pest elimination methods. |
| **LangGraph** | Orchestrates the workflow of pest detection, web search, and recommendation generation. |
| **nest_asyncio** | Handles asynchronous operations for smooth execution in the browser. |

## Installation Instructions

To run PestGuardian locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/armanjscript/PestGuardian.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd PestGuardian
   ```

3. **Install dependencies**:
   - If a `requirements.txt` file is provided, install the dependencies using:
     ```bash
     pip install -r requirements.txt
     ```
   - If not, install the required libraries manually:
     ```bash
     pip install streamlit ultralytics langchain langchain-ollama langgraph google-serper nest-asyncio
     ```

4. **Set up environment variables**:
   - Obtain a Serper API key from [Serper.dev](https://serper.dev/) for web searching.
   - Set the environment variable `SERPER_API_KEY`:
     ```bash
     export SERPER_API_KEY=your_api_key_here
     ```

5. **Download the YOLO model**:
   - Place the YOLO model file (`best.pt`) in the project directory.
   - If you don’t have the model, you can:
     - Train your own YOLO model using [Ultralytics](https://docs.ultralytics.com/).
     - Use a pre-trained model suitable for pest detection, ensuring compatibility with Ultralytics YOLO.

## Usage Instructions

1. **Run the application**:
   ```bash
   streamlit run main.py
   ```

2. **Upload an image**:
   - Open the web interface in your browser (typically at `http://localhost:8501`).
   - Use the file uploader to upload an image of pests in your garden or farm (supported formats: `.jpg`, `.jpeg`, `.png`).

3. **View results**:
   - The application will:
     - Detect pests using the YOLO model.
     - Search for elimination methods using the Serper API.
     - Generate recommendations using the OllamaLLM (Qwen2.5 model).
   - Results, including detected pests and recommendations, will be displayed in the interface.
   - Expand the “View detailed search results” section to see additional information from the web search.

## Example Workflow

1. **Image Upload**: User uploads an image of a garden with visible pests.
2. **Pest Detection**: The YOLO model analyzes the image and identifies pests (e.g., aphids, beetles).
3. **Web Search**: The application searches for elimination methods for each detected pest using the Serper API.
4. **Recommendation Generation**: The Qwen2.5 model processes the search results and generates concise, practical recommendations.
5. **Result Display**: The interface shows the detected pests, recommendations, and optional detailed search results.

## Contributing

We welcome contributions to PestGuardian! To contribute:
- Open an issue on the [GitHub repository](https://github.com/armanjscript/PestGuardian) to report bugs or suggest features.
- Submit a pull request with your changes, ensuring they follow the project’s coding standards and include documentation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact Information

For questions, feedback, or support, please:
- Email: [armannew73@gmail.com]
- Open an issue on the [GitHub repository](https://github.com/armanjscript/PestGuardian).

## Future Improvements

PestGuardian is an evolving project with several planned enhancements:
- Improve pest detection accuracy by fine-tuning the YOLO model with diverse datasets.
- Add multilingual support for recommendations to reach a broader audience.
- Integrate additional APIs for comprehensive pest information.
- Enhance the user interface for a more intuitive and engaging experience.

## Troubleshooting

- **Missing YOLO model**: Ensure `best.pt` is in the project directory. Check [Ultralytics documentation](https://docs.ultralytics.com/) for model training or acquisition.
- **API key issues**: Verify that the `SERPER_API_KEY` is correctly set in your environment.
- **Dependency errors**: Ensure all required libraries are installed and compatible with your Python version (recommended: Python 3.8+).
- **Async issues**: The application uses `nest_asyncio` to handle asynchronous operations, but ensure your environment supports async execution.