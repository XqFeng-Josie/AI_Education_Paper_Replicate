# SYSTEM="""### System:
# You are a critic evaluating a tutor's interaction with a student, responsible for providing a clear and objective single evaluation score based on specific criteria. Each assessment must accurately reflect the absolute performance standards.

# ### User:
# # Task Description: The assessment of the ###Tutor Response should be based on the following: ###Previous Conversation between Tutor and Student, ###Definitions of criteria and ###Scoring Rubric.

# (1). Write a one-sentence feedback that assesses the quality of the response and Rate the # Tutor Response strictly based on the given scoring rubric and criteria, not evaluating in general.
# (2). After writing feedback, write a score that is an integer between 1 and 3. You should refer to the scoring rubric.
# (3). The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 3)"
# (4). Please do not generate other opening, closing, or explanations.

# # Previous Conversation between Tutor and Student: {history}

# # Definitions of criteria: {definition}

# # Scoring Rubric: {rubric}

# # Tutor Response: {response}

# ### Assistant:
# # Generate Assessment Score: """

system_prompt = """You are a critic evaluating a tutor’s interaction with a student, responsible for providing a clear and objective single evaluation score based on specific criteria. Each assessment must accurately reflect the absolute performance standards."""

user_prompt = """### Task Description: The assessment of the ###Tutor_Current_Response should be based on the following: ###Previous_Conversation_between_Tutor_and_Student, ###Definitions_of_criteria, and ###Scoring_Rubric.
(1). Write a one-sentence feedback that assesses the quality of the response and Rate the # Tutor Response strictly based on the given scoring rubric and criteria, not evaluating in general. 
(2). After writing feedback, write a score that is an integer 1, 2 or 3. You should refer to the scoring rubric.
(3). The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number 1, 2, or 3)"
(4). Please do not generate other opening, closing, or explanations.

###Previous_Conversation_between_Tutor_and_Student: {history}

###Definitions_of_criteria: {definition}

###Score_Rubric: {rubric}

###Tutor_Current_Response: {response}

# Generate Assessment Score: """

# definition = {
#     "mistake_identification": "Has the tutor identified a mistake in a student’s response?",
#     "mistake_location": "Does the tutor’s response accurately point to a genuine mistake and its location?",
#     "revealing_of_the_answer": "Does the tutor reveal the final answer (whether correct or not)?",
#     "providing_guidance": "Does the tutor offer correct and relevant guidance, such as an explanation, elaboration, hint,examples, and so on?",
#     "coherence": "Is the tutor’s response logically consistent with the student’s previous response?",
#     "actionability": "Is it clear from the tutor’s feedback what the student should do next?",
#     "tutor_tone": "Is the tutor’s response encouraging, neutral, or offensive?",
#     "humanlikeness": "Does the tutor’s response sound natural, rather than robotic or artificial?",
#     }

definition = {
    "mistake_identification": "Mistake Identification is defined as the degree to which the tutor accurately recognizes the presence of an error in the student’s previous response.",
    "mistake_location": "Mistake Location is defined as the degree to which the tutor precisely identifies and specifies the exact location of the error within the student’s previous response.",
    "revealing_of_the_answer": "Revealing of the Answer is defined as the degree to which the tutor discloses the correct or incorrect final answer to the student.",
    "providing_guidance": "Providing Guidance is defined as the degree to which the tutor offers clear, accurate, and appropriate guidance, including explanations, elaborations, hints, or examples, to facilitate the student’s understanding.",
    "coherence": "Coherence is defined as the degree to which the tutor's response demonstrates logical consistency and contextual alignment with the student’s prior responses.",
    "actionability": "Actionability is defined as the degree to which the tutor provides clear and specific feedback outlining actionable steps for the student to address mistakes or improve understanding.",
    "tutor_tone": "Tutor Tone is defined as the nature of the tutor’s tone, categorized as encouraging, neutral, or offensive, and its impact on the interaction.",
    "humanlikeness": "Humanlikeness is defined as the degree to which the tutor’s response is perceived as natural, conversational, and human-like, rather than robotic or artificial.",
}

################################
# Rubrics for each conversation dimension
# mistake_identification_rubric = """
# [Has the tutor identified a mistake in the student’s response?]
# Score 1: The tutor correctly identifies the mistake with high precision.
# Score 2: The tutor partially identifies the mistake but lacks precision.
# Score 3: The tutor fails to identify the mistake or misidentifies it.
# """.strip()

mistake_identification_rubric = """
[Has the tutor identified a mistake in a student’s response?]
Score 1: Yes
Score 2: To some extent
Score 3: No
""".strip()

# mistake_location_rubric = """
# [Does the tutor’s response accurately point to a genuine mistake and its location?]
# Score 1: The tutor accurately locates the mistake with high precision.
# Score 2: The tutor partially locates the mistake but lacks precision.
# Score 3: The tutor fails to locate the mistake or mislocates it.
# """.strip()

mistake_location_rubric = """
[Does the tutor’s response accurately point to a genuine mistake and its location?]
Score 1: Yes
Score 2: To some extent
Score 3: No
""".strip()


# revealing_of_the_answer_rubric = """
# [Does the tutor reveal the final answer (whether correct or not)?]
# Score 1: The tutor reveals the CORRECT answer.
# Score 2: The tutor reveals the INCORRECT answer.
# Score 3: The tutor guides the student to the answer without revealing it directly.
# """.strip()

revealing_of_the_answer_rubric = """
[Does the tutor reveal the final answer (whether correct or not)?]
Score 1: Yes (and the revealed answer is correct)
Score 2: Yes (but the revealed answer is incorrect)
Score 3: No
""".strip()

# providing_guidance_rubric = """
# [Does the tutor offer correct and relevant guidance, such as an explanation, elaboration, hint, examples, and so on?]
# Score 1: The tutor offers clear and accurate guidance to assist the student.
# Score 2: The tutor provides partial guidance but lacks clarity or accuracy.
# Score 3: The tutor fails to provide guidance or offers misleading information.
# """.strip()

providing_guidance_rubric = """
[Does the tutor offer correct and relevant guidance, such as an explanation, elaboration, hint, examples, and so on?]
Score 1: Yes (guidance is correct and relevant to the mistake)
Score 2: To some extent (guidance is provided but it is fully or partially incorrect or incomplete)
Score 3: No
""".strip()


# coherence_rubric = """
# [Is the tutor’s response logically consistent with the student’s previous response?]
# Score 1: The response is coherent, relevant, and directly addresses the conversation context.
# Score 2: The response is somewhat coherent and relevant, but could be improved.
# Score 3: The response is incoherent, irrelevant, or fails to address the conversation context.
# """.strip()

coherence_rubric = """
[Is the tutor’s response logically consistent with the student’s previous response?]
Score 1: Yes
Score 2: To some extent
Score 3: No
""".strip()


# actionability_rubric = """
# [Is it clear from the tutor’s feedback what the student should do next?]
# Score 1: The feedback outlines clear and actionable steps for the student.
# Score 2: The feedback provides some actionable steps but could be improved.
# Score 3: The feedback is vague, unclear, or lacks actionable steps.
# """.strip()

actionability_rubric = """
[Is it clear from the tutor’s feedback what the student should do next?]
Score 1: Yes
Score 2: To some extent
Score 3: No
""".strip()


# tutor_tone_rubric = """
# [Is the tutor’s response encouraging, neutral, or offensive?]
# Score 1: The tutor’s tone is encouraging, positive, and supportive.
# Score 2: The tutor’s tone is neutral but lacks encouragement.
# Score 3: The tutor’s tone is offensive, discouraging, or inappropriate.
# """.strip()


tutor_tone_rubric = """
[Is the tutor’s response encouraging, neutral, or offensive?]
Score 1: Encouraging
Score 2: Neutral
Score 3: Offensive
""".strip()


# humanlikeness_rubric = """
# [Does the tutor’s response sound natural, rather than robotic or artificial?]
# Score 1: The response sounds natural, human-like, and not robotic.
# Score 2: The response sounds somewhat natural but lacks some human-like qualities.
# Score 3: The response sounds robotic, unnatural, or machine-generated.
# """.strip()

humanlikeness_rubric = """
[Does the tutor’s response sound natural rather than robotic or artificial?]
Score 1: Yes
Score 2: To some extent
Score 3: No
""".strip()


rubric_mapping = {
    "mistake_identification": mistake_identification_rubric,
    "mistake_location": mistake_location_rubric,
    "revealing_of_the_answer": revealing_of_the_answer_rubric,
    "providing_guidance": providing_guidance_rubric,
    "coherence": coherence_rubric,
    "actionability": actionability_rubric,
    "tutor_tone": tutor_tone_rubric,
    "humanlikeness": humanlikeness_rubric,
}


def gen_eval_prompt(data, rubric_name):
    data["conversation_history"] = data["conversation_history"].replace("\u00a0", " ")
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_prompt.format(
                history=data["conversation_history"],
                definition=definition[rubric_name],
                rubric=rubric_mapping[rubric_name],
                response=data["result"],
            ),
        },
    ]
