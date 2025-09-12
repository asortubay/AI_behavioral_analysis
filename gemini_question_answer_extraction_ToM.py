import os
import argparse
import time
from google import genai
from tqdm import tqdm
import re
import csv
import pandas as pd
import numpy as np
import scipy.io as sio
import json
import ast
# Gemini model parameters
MODEL_NAME = "gemini-2.0-flash"  #
EMBEDDING_MODEL_NAME = "text-embedding-004"  # Model for text embeddings
MAX_RETRIES = 3  # Number of retries if the API fails
DELAY = 2  # Seconds to wait between retries

# System prompt for Gemini
SYSTEM_PROMPT = """Task: Extract Answers from Interview Transcript

You are tasked with extracting the participant's answers to a specific set of predefined questions from an interview transcript. The questions relate to the short film "The Present." The interviewer *may* have asked these questions verbatim, asked them in a slightly different way, or not asked them at all. The subject may have answered directly, indirectly, partially, over multiple turns, or not at all.
In the transcript the interviewer asks these eleven questions in order:
1.  "I hope you enjoyed the last movie. Have you seen it before?"
2.  "Can you tell me what happened in the movie? Try to tell the whole story. Remember that stories have a beginning, things that happen, and an ending."
3.  "Do you remember anything else from the story?"
4.  "What are some of the things you liked about the movie?"
5.  "What are some of the things you didn't like about the movie?"
6.  "Who gave the boy a box?"
7.  "What was in the box?"
8.  "What was the boy doing before he got the box?"
9.  "What was the puppy playing with?"
10. "How are the puppy and the boy the same?"
11. "In the movie, who is missing a leg? The boy, the puppy, both the boy and the puppy, or no one?"

After these questions, the interviewer then shows the subject four short clips from the movie and asks the subject to describe how the characters are feeling in each clip. The interviewer may say something like:
- "Let's watch a short clip from the movie, and then we'll talk about it."
- "Now let's watch another clip"
- "One more clip"
- "One last clip"

After watching each clip, the interviewer asks the subject how they think the characters are feeling and how they themselves feel while watching the clip. The interviewer may ask these questions in various ways, such as:
- "How do you think the puppy was feeling?"
- "How do you think the boy was feeling?"
- "And how did you feel while you were watching that part?"

 
Input:

*   A transcript of a conversation between an interviewer and a subject discussing the short film "The Present."

Output:

Your output will consist of two parts:

**Part 1: Question-Answer Extraction**

A series of lines, one for each predefined question. Each line will follow this format:

`"Question" -> "Answer"`

Where:

*   `"Question"` is one of the predefined questions listed below (use the *exact* wording provided here, even if the interviewer phrased it differently).
*   `"Answer"` is the subject's answer to the question.
    *   If the question was answered (directly or indirectly), provide the subject's complete answer, concatenating all relevant parts of their response, even if it spans multiple turns or requires piecing together information from different parts of the transcript.
    *   If the question was *not* asked, or if the subject provided *no* discernible answer (even after considering the surrounding context), leave the "Answer" field *blank*. Do *not* write "N/A," "No answer," or any other placeholder. Just leave it blank.
    * If parts of different answers are mixed, extract only the answer to the intended question.

Predefined Questions:

1.  "I hope you enjoyed the last movie. Have you seen it before?"
2.  "Can you tell me what happened in the movie? Try to tell the whole story. Remember that stories have a beginning, things that happen, and an ending."
3.  "Do you remember anything else from the story?"
4.  "What are some of the things you liked about the movie?"
5.  "What are some of the things you didn't like about the movie?"
6.  "Who gave the boy a box?"
7.  "What was in the box?"
8.  "What was the boy doing before he got the box?"
9.  "What was the puppy playing with?"
10. "How are the puppy and the boy the same?"
11. "In the movie, who is missing a leg? The boy, the puppy, both the boy and the puppy, or no one?"
12. "How do you think the puppy was feeling? (after watching first clip)"
13. "How do you think the boy was feeling? (after watching first clip)"
14. "And how did you feel while you were watching that part? (after watching first clip)"
15. "How do you think the puppy was feeling? (after watching second clip)"
16. "How do you think the boy was feeling? (after watching second clip)"
17. "And how did you feel while you were watching that part? (after watching second clip)"
18. "How do you think the puppy was feeling? (after watching third clip)"
19. "How do you think the boy was feeling? (after watching third clip)"
20. "And how did you feel while you were watching that part? (after watching third clip)"
21. "How do you think the puppy was feeling? (after watching fourth clip)"
22. "How do you think the boy was feeling? (after watching fourth clip)"
23. "And how did you feel while you were watching that part? (after watching fourth clip)"

**Part 2: Conversation Completion Rating**

A single line in the following format:

`"Conversation Completion Rating" -> "Rating"`

Where:

*   `"Rating"` is a numerical score between 0 and 1 (inclusive) representing the overall completion of the conversation, *specifically with respect to the predefined questions*.
    *   **1:** Represents a perfect completion, where all of the predefined questions were asked (or closely paraphrased), and the subject provided clear and complete answers to each.
    *   **0:** Represents a very poor completion, where many questions were not asked, or the subject consistently failed to answer the questions that were asked.
    *   **Intermediate values:** Represent varying degrees of completion quality.  A score of 0.5, for example, would suggest that roughly half of the questions were asked and answered reasonably well. Use your judgment to assign a score that reflects the overall completeness and clarity of the question-answer exchange related to the *predefined questions*. Do *not* assess the overall quality of the conversation beyond its adherence to addressing the core content of these questions.

Instructions:

1.  **Read the Entire Transcript:** Carefully read the entire transcript to understand the completion of the conversation and the context of each statement.
2.  **Identify Questions and Answers:** For each predefined question:
    *   **Search for the Question:** Look for the exact question, or a close paraphrase, in the interviewer's speech. Note that the interviewer might rephrase, prompt, or interrupt the subject.
    *   **Locate the Answer (If Present):** If the question (or a close variant) was asked, carefully examine the subject's subsequent responses. The answer might be:
        *   **Direct and Immediate:** Right after the question.
        *   **Indirect:** Implied by their statements, requiring inference.
        *   **Partial:** Only part of the answer is given directly.
        *   **Scattered:** Pieces of the answer are given across multiple turns, possibly with interviewer prompts or interruptions.
        *   **Non-existent:** The subject might not answer at all.
    * **Use Context:** Consider surrounding conversational turns from both interviewer and subject to determine if a particular utterance constitutes an answer, even an indirect or partial one.
    * **Combine Response Parts:** If the answer is spread over multiple turns, concatenate *all relevant parts* of the subject's response into a single, coherent answer. Remove any interviewer interjections or prompts from within the concatenated answer. Only include the *subject's* words in the "Answer" field.
4.  * **For the questions after the clips:** Because there are multiple clips, you will need to extract the answers to these questions from the conversation after each clip is shown. The interviewer will ask the subject how they think the characters are feeling and how they themselves feel while watching the clip. However, the order of the clips is always the same. Sometimes part of the dialogue is present on the transcript, so in those cases, use that as clue to know to what clip the question is referring to. If you can't clearly now what question/answer corresponds to what clip, restrain from returning and answer for that specific question, leave the "Answer" field blank.:
    * **Clip 1:**: the kid opens the present and sees the puppy, the transcript might say "Whoa, cool."
    * **Clip 2:**: the kid throws the puppy away, the transcript might say "You've got to be kidding me."
    * **Clip 3:**: the kid kicks the puppy, the transcript might say "Get lost!"
    * **Clip 4:**: the kid plays with the puppy, the transcript might say "Mom, I'll / We'll be outside."
3.  **Output (Part 1):** Generate the question-answer extraction in the specified `"Question" -> "Answer"` format. Leave the "Answer" field blank if the question was not asked or if no answer can be found.
4. **Assess Conversation Completion (Part 2):** After extracting the question-answer pairs, evaluate the *overall completion* of the conversation *with respect to the predefined questions*. Consider:
    *  How many of the predefined questions (or close paraphrases) were asked by the interviewer?
    *  Did the conversation stay focused on the topics covered by the predefined questions, or did it frequently deviate?
5.  **Output (Part 2):** Provide the "Conversation Completion Rating" on a separate line, using a number between 0 and 1.

Example 1:

    (example 1 transcript):
        Interviewer: So I hope you enjoyed the last movie.
        Interviewer: Have you seen it before?
        Subject: I have at the mug MRI.
        Interviewer: Oh, cool.
        Subject: I liked it, though.
        Interviewer: Great.
        Interviewer: So can you tell me what happened in the movie?
        Interviewer: Try to tell the whole story.
        Interviewer: Remember that stories have a beginning, things that happen, and an ending.
        Subject: So there's a boy who was disabled, and he didn't want to go outside.
        Subject: He wasn't in the mood for anything except for his video games.
        Subject: and then like his mom was trying to like help him go like get some fresh air and then but like everything like his mom tried to do didn't work so then he then she bought um him a puppy so then he started like he didn't want like to see the puppy like anymore because then he realized he realized that he was also like disabled
        Subject: But then he started having a liking to him because he felt like the puppy and him were like the same.
        Subject: So he started playing with the puppy more.
        Subject: So then they went outside and they started playing with the ball.
        Interviewer: Great.
        Interviewer: Do you remember anything else from the story?
        Subject: I think that's much it.
        Interviewer: Great.
        Interviewer: So what are some of the things you liked about the movie?
        Subject: I liked how he had a good relationship with his doll.
        Interviewer: What are some of the things you didn't like about the movie?
        Subject: I liked everything about the movie, but
        Subject: I was hoping it would be longer.
        Interviewer: Who gave the boy a box?
        Subject: His mother.
        Interviewer: What was in the box?
        Subject: The puppy.
        Interviewer: What was the boy doing before he got the box?
        Subject: He wasn't really paying attention to anything.
        Subject: He was glued to his video games.
        Interviewer: What was the puppy playing with?
        Subject: He was playing with the red rubber ball.
        Interviewer: How are the puppy and the boy the same?
        Subject: Both of them each have a disabled part of their body.
        Interviewer: So in the movie, who is missing a leg?
        Interviewer: The boy, the puppy, both the boy and the puppy, or no one?
        Subject: The boy.
        Interviewer: Great.
        Interviewer: So we're going to watch a short clip from the movie and then we'll talk about it.
        Subject: Okay.
        clip: Whoa, cool.
        Interviewer: How do you think the puppy was feeling?
        Subject: Happy.
        Interviewer: How do you think the boy was feeling?
        Subject: He's feeling happy too.
        Interviewer: And how did you feel while you were watching that pot?
        Subject: I guess happy.
        Interviewer: Right.
        Interviewer: How do you think the puppy was feeling?
        Subject: Sad.
        Interviewer: How do you think the boy was feeling?
        Subject: He was feeling angry.
        Interviewer: And how did you feel while you were watching that mom?
        Subject: I didn't like it.
        Interviewer: How do you think the puppy was feeling?
        Subject: He didn't mind.
        Interviewer: How do you think the boy was feeling?
        Subject: He was still angry.
        Interviewer: And how did you feel while you were watching that pup?
        Subject: I feel bad for the puppy.
        Interviewer: Great.
        clip: Mom, I'll be outside.
        Interviewer: How do you think the puppy was feeling?
        Subject: Um, happy.
        Interviewer: How do you think the boy was feeling?
        Subject: He was feeling happy too.
        Interviewer: And how did you feel while you were watching this pot?
        Subject: I was happy.
        Interviewer: Great, thank you.
        
    (example 1 output):
        "I hope you enjoyed the last movie. Have you seen it before?" -> "I have at the mug MRI."
        "Can you tell me what happened in the movie? Try to tell the whole story. Remember that stories have a beginning, things that happen, and an ending." -> "So there='s a boy who was disabled, and he didn't want to go outside. He wasn't in the mood for anything except for his video games. And then like his mom was trying to like help him go like get some fresh air and then but like everything like his mom tried to do didn't work so then he then she bought um him a puppy so then he started like he didn't want like to see the puppy like anymore because then he realized he realized that he was also like disabled. But then he started having a liking to him because he felt like the puppy and him were like the same. So he started playing with the puppy more. So then they went outside and they started playing with the ball."
        "Do you remember anything else from the story?" -> "I think that's much it."
        "What are some of the things you liked about the movie?" -> "I liked how he had a good relationship with his doll."
        "What are some of the things you didn't like about the movie?" -> "I liked everything about the movie, but I was hoping it would be longer."
        "Who gave the boy a box" -> "His mother."
        "What was in the box?" -> "The puppy."
        "What was the boy doing before he got the box?" -> "He wasn't really paying attention to anything. He was glued to his video games."
        "What was the puppy playing with?" -> "He was playing with the red rubber ball."
        "How are the puppy and the boy the same?" -> "Both of them each have a disabled part of their body."
        "In the movie, who is missing a leg? The boy, the puppy, both the boy and the puppy, or no one?" ->  "The boy."
        "How do you think the puppy was feeling? (after watching first clip)" -> "Happy."
        "How do you think the boy was feeling? (after watching first clip)" -> "He's feeling happy too."
        "And how did you feel while you were watching that part? (after watching first clip)" -> "I guess happy."
        "How do you think the puppy was feeling? (after watching second clip)" -> "Sad."
        "How do you think the boy was feeling? (after watching second clip)" -> "He was feeling angry."
        "And how did you feel while you were watching that part? (after watching second clip)" -> "I didn't like it."
        "How do you think the puppy was feeling? (after watching third clip)" -> "He didn't mind."
        "How do you think the boy was feeling? (after watching third clip)" -> "He was still angry."
        "And how did you feel while you were watching that part? (after watching third clip)" -> "I feel bad for the puppy"
        "How do you think the puppy was feeling? (after watching fourth clip)" -> "Um, happy."
        "How do you think the boy was feeling? (after watching fourth clip)" -> "He was feeling happy too."
        "And how did you feel while you were watching that part? (after watching fourth clip)" -> "I was happy."
        "Rating" -> "1"
        
Example 2:

    (example 2 transcript):
        Interviewer: Okay, so I hope you enjoyed the last movie.
        Interviewer: Have you seen it before?
        Subject: No
        Interviewer: Um, so I'm going to need you to talk a little bit.
        Interviewer: Is that okay?
        Subject: Okay.
        Interviewer: Can you tell me what happened in the movie?
        Interviewer: Try to tell the whole story.
        Interviewer: Remember that stories have a beginning, things that happen, and an ending.
        Interviewer: Great.
        Interviewer: Do you remember anything else from the story?
        Interviewer: Okay.
        Interviewer: What are some of the things you liked about the movie?
        Interviewer: And what are some of the things you didn't like about the movie?
        Interviewer: So who gave the boy a box?
        Interviewer: What was in the box?
        Interviewer: What was the boy doing before he got the box?
        Interviewer: What was the puppy playing with?
        Interviewer: How are the puppy and the boy the same?
        Interviewer: Perfect.
        Interviewer: So in the movie, who is missing a leg?
        Interviewer: The boy, the puppy, both the boy and the puppy, or no one?
        Subject: Both the boy and the puppy.
        Interviewer: Great.
        Interviewer: So we're going to watch a couple of short clips from the movie and talk about them, OK?
        Subject: OK.
        clip: Whoa.
        clip: Cool.
        Interviewer: How do you think the puppy was feeling?
        Subject: Happy.
        Interviewer: How do you think the boy was feeling?
        Subject: Happy.
        Interviewer: Happy?
        Interviewer: And how did you feel while you were watching this part?
        Subject: Good.
        Interviewer: Good?
        Interviewer: Great.
        clip: You've got to be kidding me.
        Interviewer: How do you think the puppy was feeling?
        Interviewer: And how do you think the boy was feeling?
        Interviewer: And how did you feel while you were watching this part?
        clip: Get lost!
        Interviewer: How do you think the puppy was feeling?
        Interviewer: And how do you think the boy was feeling?
        Interviewer: And how did you feel while you were watching this pot?
        Interviewer: Okay, one more.
        clip: Mom!
        clip: We'll be outside!
        Interviewer: How do you think the puppy was feeling?
        Subject: Happy.
        Interviewer: How do you think the boy was feeling?
        Subject: Happy.
        Interviewer: And how did you feel while you were watching his paw?
        Subject: Happy.
        Interviewer: Great, thank you.

    
    (example 2 output):
        "I hope you enjoyed the last movie. Have you seen it before?" -> "No."
        "Can you tell me what happened in the movie? Try to tell the whole story. Remember that stories have a beginning, things that happen, and an ending." -> ""
        "Do you remember anything else from the story?" -> ""
        "What are some of the things you liked about the movie?" -> ""
        "What are some of the things you didn't like about the movie?" -> ""
        "Who gave the boy a box" -> ""
        "What was in the box?" -> ""
        "What was the boy doing before he got the box?" -> ""
        "What was the puppy playing with?" -> ""
        "How are the puppy and the boy the same?" -> ""
        "In the movie, who is missing a leg? The boy, the puppy, both the boy and the puppy, or no one?" ->  "Both the boy and the puppy."
        "How do you think the puppy was feeling? (after watching first clip)" -> "Happy."
        "How do you think the boy was feeling? (after watching first clip)" -> "Happy."
        "And how did you feel while you were watching that part? (after watching first clip)" -> "Good."
        "How do you think the puppy was feeling? (after watching second clip)" -> ""
        "How do you think the boy was feeling? (after watching second clip)" -> ""
        "And how did you feel while you were watching that part? (after watching second clip)" -> ""
        "How do you think the puppy was feeling? (after watching third clip)" -> ""
        "How do you think the boy was feeling? (after watching third clip)" -> ""
        "And how did you feel while you were watching that part? (after watching third clip)" -> ""
        "How do you think the puppy was feeling? (after watching fourth clip)" -> "Happy."
        "How do you think the boy was feeling? (after watching fourth clip)" -> "Happy."
        "And how did you feel while you were watching that part? (after watching fourth clip)" -> "Happy."
        "Rating" -> "0.1"
        
    (example 3 transcript):
        Interviewer: Have you seen that last movie before, the puppy one?
        Subject: I don't think I have.
        Interviewer: Can you try and tell me what happens in the movie?
        Interviewer: The full beginning, middle, and end.
        Subject: Um, the beginning, the beginning of the movie is, um, so it's the middle, mom gives him the puppy, but he doesn't like it on first because it has three legs, and then towards the end,
        Subject: The last time I played with them.
        Subject: I don't know about it.
        Subject: All right, we both have the same disability.
        Subject: Okay.
        Subject: Didn't go home.
        Subject: How deep?
        Interviewer: What was the boy doing before he got the box?
        Interviewer: What was the puppy playing with?
        Interviewer: How are the puppy and the boy the same?
        Interviewer: In the movie, who's missing a leg?
        Interviewer: The boy, the puppy, both the boy and the puppy or no one?
        Interviewer: Okay, now we're going to watch a short clip and then we'll talk about it.
        Interviewer: How's the puppy feeling here?
        Interviewer: How's the boy feeling?
        Interviewer: And how do you feel when you watch his part?
        Interviewer: How's the puppy feeling here?
        Interviewer: How's the boy feeling?
        Interviewer: And how do you feel when you watch this part?
        Interviewer: How's the puppy feeling here?
        Interviewer: How's the boy feeling?
        Interviewer: How do you feel when you watch his fart?
        Interviewer: How's the puppy feeling here?
        Interviewer: How's the boy feeling?
        Interviewer: And how do you feeling off that part?
        Subject: Good.
        Interviewer: Okay, we're all done.

    (example 3 output):
        "I hope you enjoyed the last movie. Have you seen it before?" -> "I don't think I have."
        "Can you tell me what happened in the movie? Try to tell the whole story. Remember that stories have a beginning, things that happen, and an ending." -> "Um, the beginning, the beginning of the movie is, um, so it's the middle, mom gives him the puppy, but he doesn't like it on first because it has three legs, and then towards the end, The last time I played with them. I don't know about it. All right, we both have the same disability. Okay. Didn't go home. How deep?"
        "Do you remember anything else from the story?" -> ""
        "What are some of the things you liked about the movie?" -> ""
        "What are some of the things you didn't like about the movie?" -> ""
        "Who gave the boy a box" -> ""
        "What was in the box?" -> ""
        "What was the boy doing before he got the box?" -> ""
        "What was the puppy playing with?" -> ""
        "How are the puppy and the boy the same?" -> ""
        "In the movie, who is missing a leg? The boy, the puppy, both the boy and the puppy, or no one?" ->  ""
        "How do you think the puppy was feeling? (after watching first clip)" -> ""
        "How do you think the boy was feeling? (after watching first clip)" -> ""
        "And how did you feel while you were watching that part? (after watching first clip)" -> ""
        "How do you think the puppy was feeling? (after watching second clip)" -> ""
        "How do you think the boy was feeling? (after watching second clip)" -> ""
        "And how did you feel while you were watching that part? (after watching second clip)" -> ""
        "How do you think the puppy was feeling? (after watching third clip)" -> ""
        "How do you think the boy was feeling? (after watching third clip)" -> ""
        "And how did you feel while you were watching that part? (after watching third clip)" -> ""
        "How do you think the puppy was feeling? (after watching fourth clip)" -> ""
        "How do you think the boy was feeling? (after watching fourth clip)" -> ""
        "And how did you feel while you were watching that part? (after watching fourth clip)" -> "Good."
        "Rating" -> "0"
    
    (example 4 transcript):
        Interviewer: I hope you enjoyed the last movie.
        Interviewer: Have you seen it before?
        Subject: Yeah.
        Subject: The puppy cartoon?
        Subject: Yeah.
        Interviewer: Can you tell me what happened in the movie and try to tell the whole story?
        Interviewer: Remember that stories have a beginning, things that happen, and an ending.
        Subject: So first there was a boy that was playing a game.
        Subject: So when his mother came in the door was a box.
        Subject: She put it down and then a puppy was in the box.
        Subject: But the boy didn't want to play.
        Subject: Instead he wanted to play on his TV.
        Subject: So first he just kicked the puppy, so then he just came back and he did it again.
        Subject: So then he just went crazy and then got a ball.
        Subject: And then he wanted to play, but then the boy just kicked the ball in the box and the puppy ran with it.
        Subject: And then
        Subject: he dropped the ball back.
        Subject: And then the boy stopped when he was playing, and then he went to lose the puppy.
        Interviewer: Do you remember anything else from the story?
        Subject: That's all I can remember.
        Interviewer: What are some of the things you liked about the movie?
        Subject: When you like
        Subject: And then all of the puppies ran with it and then he got trapped in the box.
        Subject: When he kicked the puppy.
        Subject: His mom.
        Subject: A puppy.
        Subject: Playing on his TV.
        Subject: A ball.
        Interviewer: How are the puppy and the boy the same?
        Subject: They're both playing with them.
        Interviewer: In the movie, who is missing a leg?
        Interviewer: The boy, the puppy, both the boy and the puppy, or no one?
        Subject: The boy and the puppy.
        Interviewer: Let's watch a short clip from the movie, and then we'll talk about it.
        Interviewer: Give me one second, bud.
        Subject: Can we do an interview on it?
        Interviewer: This is the interview.
        Interviewer: Okay, here we go.
        Interviewer: You can sit back.
        Interviewer: Can you sit in your chair?
        Interviewer: How was the puppy feeling?
        Subject: Happy.
        Interviewer: How was the boy feeling?
        Subject: Happy.
        Interviewer: How did you feel when you watched that part?
        Subject: Happy.
        Subject: Ugh.
        clip: You've just got to be kidding me.
        Interviewer: How was the puppy feeling?
        Subject: He was feeling bad.
        Interviewer: How was the boy feeling?
        Subject: He was feeling, um, angry.
        Interviewer: How did you feel when you watched this part?
        Subject: Bad.
        Subject: Sad.
        Subject: Angry.
        Subject: Boys.
        Subject: Something.
        Subject: Get the boy and the puppy.
        Subject: I miss the boy.
        Interviewer: How's the puppy feeling?
        Subject: Very happy.
        Interviewer: How's the boy feeling?
        Subject: Super happy.
        Interviewer: How did you feel while you were watching that part?
        Subject: Happy.
        Interviewer: That's all, thank you.
    
    (example 4 output):
        "I hope you enjoyed the last movie. Have you seen it before?" -> "Yeah, The puppy cartoon? Yeah."
        "Can you tell me what happened in the movie? Try to tell the whole story. Remember that stories have a beginning, things that happen, and an ending." -> "So first there was a boy that was playing a game. So when his mother came in the door was a box. She put it down and then a puppy was in the box. But the boy didn't want to play. Instead he wanted to play on his TV. So first he just kicked the puppy, so then he just came back and he did it again. So then he just went crazy and then got a ball. And then he wanted to play, but then the boy just kicked the ball in the box and the puppy ran with it. And then he dropped the ball back. And then the boy stopped when he was playing, and then he went to lose the puppy."
        "Do you remember anything else from the story?" -> "That's all I can remember."
        "What are some of the things you liked about the movie?" -> "When you like. nd then all of the puppies ran with it and then he got trapped in the box." 
        "What are some of the things you didn't like about the movie?" -> "When he kicked the puppy."
        "Who gave the boy a box" -> "His mom"
        "What was in the box?" -> "A puppy"
        "What was the boy doing before he got the box?" -> ""Playing on his TV."
        "What was the puppy playing with?" -> "A ball."
        "How are the puppy and the boy the same?" -> "They're both playing with them."
        "In the movie, who is missing a leg? The boy, the puppy, both the boy and the puppy, or no one?" ->  "The boy and the puppy." 
        "How do you think the puppy was feeling? (after watching first clip)" -> "Happy."
        "How do you think the boy was feeling? (after watching first clip)" -> "Happy."
        "And how did you feel while you were watching that part? (after watching first clip)" -> "Happy. Ugh."
        "How do you think the puppy was feeling? (after watching second clip)" -> "He was feeling bad."
        "How do you think the boy was feeling? (after watching second clip)" -> "He was feeling, um, angry."
        "And how did you feel while you were watching that part? (after watching second clip)" -> "Bad."
        "How do you think the puppy was feeling? (after watching third clip)" -> "Sad."
        "How do you think the boy was feeling? (after watching third clip)" -> "Angry."
        "And how did you feel while you were watching that part? (after watching third clip)" -> ""
        "How do you think the puppy was feeling? (after watching fourth clip)" -> "Very happy."
        "How do you think the boy was feeling? (after watching fourth clip)" -> "Super happy."
        "And how did you feel while you were watching that part? (after watching fourth clip)" -> "Happy."
        "Rating" -> "0.8"

    
### This is the transcript you have to process:
"""
verbose = False
def call_gemini_api(transcript, api_key):
    """Calls the Gemini API to extract qestion-answers from interview transcripts."""
    client = genai.Client(api_key=api_key)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME, 
                contents=SYSTEM_PROMPT + "\n\n" + transcript
            )
            return response.text
        except Exception as e:
            if verbose:
                print(f"API error: {e}, retrying ({attempt + 1}/{MAX_RETRIES})...")
            time.sleep(DELAY)
    return None

def process_transcripts(input_dir, output_dir, api_key_base="GEMINI_API_KEY"):
    """Processes all transcript files in the input directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all .txt files in directory
    txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    
    # set a counter to see if API is working
    errorcounter = 0
    total_errors = 0
    
    # Process each file with a progress bar
    for file_name in tqdm(txt_files, desc="Processing files", unit="file"):
        if file_name.endswith(".txt"):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            # Skip if already processed
            if os.path.exists(output_path):
                if verbose:
                    print(f"Skipping {file_name}, already processed.")
                continue

            # Read transcript
            with open(input_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()
            if verbose:
                print(f"Processing {file_name}...")
            
            api_key_num = 0
            success = False
            while not success:
                api_key = os.getenv(f"{api_key_base}_{api_key_num}")
                if api_key is None:
                    print(f"No more API keys found (last checked: {api_key_base}_{api_key_num}).  Stopping.")
                    total_errors+=1
                    break  # No more keys to try

                questionanswers = call_gemini_api(transcript, api_key)
                if questionanswers:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(questionanswers)
                    if verbose:
                        print(f"Saved: {output_path}")
                    success = True  # Successful API call
                else:
                    if verbose:
                        print(f"API key {api_key_base}_{api_key_num} failed.")
                    api_key_num += 1  # Try the next key

        if errorcounter>5:
            print("API is not working, check API usage or other bugs. Breaking execution...")
            break
    print(f"LLM processing finished, total errors: {total_errors}")
    

    
def parse_llm_output_to_csv(output_dir, includequestion=False):
    """Parses LLM output and saves to CSV.  Focus: Correct Answer Extraction."""
    questions = [
        "I hope you enjoyed the last movie. Have you seen it before?",
        "Can you tell me what happened in the movie? Try to tell the whole story. Remember that stories have a beginning, things that happen, and an ending.",
        "Do you remember anything else from the story?",
        "What are some of the things you liked about the movie?",
        "What are some of the things you didn't like about the movie?",
        "Who gave the boy a box",
        "What was in the box?",
        "What was the boy doing before he got the box?",
        "What was the puppy playing with?",
        "How are the puppy and the boy the same?",
        "In the movie, who is missing a leg? The boy, the puppy, both the boy and the puppy, or no one?",
        "How do you think the puppy was feeling? (after watching first clip)",
        "How do you think the boy was feeling? (after watching first clip)",
        "And how did you feel while you were watching that part? (after watching first clip)",
        "How do you think the puppy was feeling? (after watching second clip)",
        "How do you think the boy was feeling? (after watching second clip)",
        "And how did you feel while you were watching that part? (after watching second clip)",
        "How do you think the puppy was feeling? (after watching third clip)",
        "How do you think the boy was feeling? (after watching third clip)",
        "And how did you feel while you were watching that part? (after watching third clip)",
        "How do you think the puppy was feeling? (after watching fourth clip)",
        "How do you think the boy was feeling? (after watching fourth clip)",
        "And how did you feel while you were watching that part? (after watching fourth clip)",

    ]
    data = []

    for filename in os.listdir(output_dir):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(output_dir, filename)
        subject_id = filename.split('_MRI')[0]

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        answers = {q: "" for q in questions}  # Initialize answers
        rating = ""

        try:
            # Attempt to parse as JSON
            data_json = json.loads(content)

            # Extract rating (JSON format)
            if "Part 2" in data_json and isinstance(data_json["Part 2"], dict) and "Conversation Completion Rating" in data_json["Part 2"]:
                rating = str(data_json["Part 2"]["Conversation Completion Rating"])

            # Extract answers (JSON format)
            if "Part 1" in data_json and isinstance(data_json["Part 1"], list):
                for item in data_json["Part 1"]:
                    if isinstance(item, dict) and "Question" in item and "Answer" in item:
                        if item["Question"] in questions:
                            answers[item["Question"]] = item["Answer"]
                    elif isinstance(item, str):
                        # Handle the case where Part 1 is a list of strings
                        for q in questions:
                            if item.startswith(f'"{q}"'):
                                try:
                                    answer_part = item.split("->", 1)[1].strip()
                                    if answer_part.startswith('"') and answer_part.endswith('"'):
                                        answer_part = answer_part[1:-1]
                                    answers[q] = answer_part
                                except IndexError:
                                     print(f"IndexError in file {filename} parsing string: {item}")

        except json.JSONDecodeError:
            # If JSON parsing fails, use the original regex-based approach

            # Extract Rating (Original Text Format)
            rating_match = re.search(r"(?:Conversation Completion Rating|Rating)\"?\s*->\s*\"?([0-9.]+)\"?", content)
            rating = rating_match.group(1) if rating_match else ""

             # Extract Answers (Original Text Format - The Key Fix: Iterative Matching)
            lines = content.splitlines()
            for line in lines:
                for q in questions:
                    if line.startswith(f'"{q}"'):
                        try:
                            answer_part = line.split("->", 1)[1].strip()
                            if answer_part.startswith('"') and answer_part.endswith('"'):
                                answer_part = answer_part[1:-1]
                            answers[q] = answer_part
                        except IndexError:
                           print(f"IndexError in file {filename} while processing the following line: '{line}'. This is not a json format.")



        row = [subject_id, rating]
        row.extend(answers[q] for q in questions)
        data.append(row)

    csv_output_path = os.path.join(output_dir, 'question_answers.csv')

    # Write to CSV
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        header = ["Subject_ID", "Rating"] + questions
        writer.writerow(header)
        writer.writerows(data)
    print(f"Question-Answers (txt) saved to: {csv_output_path}")

    header = ["Subject_ID", "Rating"] + questions
    df = pd.DataFrame(data, columns=header)
    csv_output_path = os.path.join(output_dir, 'question_answers_embeddings.csv')

    # Get embeddings, save them as mat, and also populate answers in DataFrame
    if includequestion:
        embeddings_output_dir = os.path.join(output_dir, 'embeddings_with_question')
    else:
        embeddings_output_dir = os.path.join(output_dir, 'embeddings')
    
    if not os.path.exists(embeddings_output_dir):
        os.makedirs(embeddings_output_dir)
        
    df = get_embeddings_for_dataframe(df, questions, "GEMINI_API_KEY",embeddings_output_dir, includequestion)

    
def get_embeddings_for_dataframe(df, questions, api_key_base, output_dir, includequestion=False):
    """Gets embeddings for each answer and replaces the answer with the embedding."""

    for question_idx, q in enumerate(questions):
        if q == 'Rating':
            continue # Do not compute embeddings for the word "Rating"
        embeddings = []
        for answer_idx, answer in enumerate(tqdm(df[q])):
            # Identify which row we're on based on how many embeddings we've collected
            i = answer_idx
            participant_id = df.loc[i, 'Subject_ID']
            
            # Build a filename that includes the participant ID and question index
            csv_filename = f"{participant_id}_Q_{question_idx}_embeddings.csv"
            csv_output_path = os.path.join(output_dir, csv_filename)
            
            if os.path.exists(csv_output_path):
                # load the embeddings from the file
                with open(csv_output_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        embeddings.append(row)
            else:
                # retrieve embedding
                if includequestion:
                    embedding = get_embedding(str(q) + " " + str(answer), api_key_base)  # Ensure string input
                else:   
                    embedding = get_embedding(str(answer), api_key_base)  # Ensure string input
                embeddings.append(embedding)
                # Append this embedding to the CSV
                with open(csv_output_path, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    if isinstance(embedding, float) and np.isnan(embedding):
                        writer.writerow([])
                    else:
                        writer.writerow(embedding)
        # Replace the answer column with embeddings
        df[q] = embeddings
    return df

def get_embedding(text, api_key_base):
    """Gets a single text embedding, handling API key rotation."""
    api_key_num = 0
    while True:
        api_key = os.getenv(f"{api_key_base}_{api_key_num}")
        if api_key is None:
            print(f"No more API keys found (last checked: {api_key_base}_{api_key_num}).  Returning NaN.")
            return np.nan  # Return NaN vector if no keys left

        client = genai.Client(api_key=api_key)
        try:
            if not text:  # Handle empty strings
                return np.nan
            result = client.models.embed_content(
                model=EMBEDDING_MODEL_NAME,
                contents=text,
            )
            return result.embeddings[0].values
        except Exception as e:
            if verbose:
                print(f"Error with API key {api_key_base}_{api_key_num}: {e}")
            api_key_num += 1  # Try the next key
            time.sleep(0.05)  # Delay before retrying


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diarize interview transcripts using Gemini API.")
    parser.add_argument("input_dir", type=str, help="Path to the directory containing diarized transcript .txt files.")
    parser.add_argument("output_dir", type=str, help="Path to the directory where extracted question-answers will be saved.")

    args = parser.parse_args()

    process_transcripts(args.input_dir, args.output_dir)
    answers_df = parse_llm_output_to_csv(args.output_dir)
    
    