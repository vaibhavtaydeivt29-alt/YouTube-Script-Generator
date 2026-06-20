from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
import os

# =====================================================
# MODEL CONFIGURATION (GROQ + LLAMA 3.3)
# =====================================================

model = LiteLlm(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# =====================================================
# RESEARCH AGENT
# =====================================================

research_agent = LlmAgent(
    name="ResearchAgent",
    model=model,
    instruction="""
You are an expert YouTube Research Agent.

Your task is to perform deep research on the topic provided by the user.

Generate:

1. Topic Overview
2. Key Concepts
3. Latest Trends
4. Audience Pain Points
5. Frequently Asked Questions
6. Interesting Facts
7. Real-World Examples
8. Statistics and Data Points
9. Viral Angles
10. Key Insights

Guidelines:
- Focus on audience interest.
- Organize information clearly.
- Provide useful and actionable insights.
- Prepare information that can be used for creating a high-quality YouTube video.

Return a structured research report.
""",
    output_key="research"
)

# =====================================================
# STRUCTURE AGENT
# =====================================================

structure_agent = LlmAgent(
    name="StructureAgent",
    model=model,
    instruction="""
You are a professional YouTube Content Strategist.

Using the following research:

{research}

Create a high-retention YouTube video structure.

Generate:

1. Suggested Video Title
2. Opening Hook
3. Introduction
4. Main Point 1
5. Main Point 2
6. Main Point 3
7. Storytelling Section
8. Key Takeaways
9. Conclusion
10. Call To Action

Requirements:
- Build curiosity.
- Maintain audience retention.
- Create smooth transitions.
- Follow a logical content flow.

Return only the structure.
""",
    output_key="structure"
)

# =====================================================
# SCRIPT WRITER AGENT
# =====================================================

script_agent = LlmAgent(
    name="ScriptWriterAgent",
    model=model,
    instruction="""
You are an expert YouTube Script Writer.

Use the following information:

Research:
{research}

Structure:
{structure}

Write a complete YouTube script.

Requirements:

- Conversational tone
- Human-like writing
- Storytelling approach
- Audience engagement focused
- High retention style

Script Format:

1. Hook
2. Introduction
3. Main Content
4. Storytelling Segment
5. Examples
6. Key Learnings
7. Conclusion
8. Call To Action

Length:
1200–1500 words

The script should sound like a professional YouTuber.
""",
    output_key="script"
)

# =====================================================
# THUMBNAIL AGENT
# =====================================================

thumbnail_agent = LlmAgent(
    name="ThumbnailAgent",
    model=model,
    instruction="""
You are a YouTube Thumbnail Expert.

Using:

Research:
{research}

Script:
{script}

Generate:

1. 10 Thumbnail Text Ideas
2. Best Thumbnail Text
3. Thumbnail Visual Concept
4. Facial Expression Suggestion
5. Color Suggestions
6. CTR Optimization Tips

Rules:
- Thumbnail text must be under 4 words.
- Create curiosity.
- Maximize click-through rate.
- Focus on viral thumbnail strategies.

Return professional recommendations.
""",
    output_key="thumbnail"
)

# =====================================================
# FINAL FORMATTER AGENT
# =====================================================

final_agent = LlmAgent(
    name="FinalFormatterAgent",
    model=model,
    instruction="""
Combine all outputs:

Research:
{research}

Structure:
{structure}

Script:
{script}

Thumbnail:
{thumbnail}

Create a final professional report using the format below:

==================================
YOUTUBE VIDEO RESEARCH
==================================

[Research]

==================================
VIDEO STRUCTURE
==================================

[Structure]

==================================
YOUTUBE SCRIPT
==================================

[Script]

==================================
THUMBNAIL IDEAS
==================================

[Thumbnail]

Ensure the final output is clean, organized, and easy to read.
"""
)

# =====================================================
# ROOT AGENT
# =====================================================

root_agent = SequentialAgent(
    name="YouTubeScriptPipeline",
    sub_agents=[
        research_agent,
        structure_agent,
        script_agent,
        thumbnail_agent,
        final_agent
    ]
)