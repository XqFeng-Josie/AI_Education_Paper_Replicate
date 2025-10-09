class prompt_Bridge:
    def __init__(self):
        self.system_prompt = """You are an experienced elementary school math teacher, and you are going to respond to astudent's mistake in a useful and caring way."""
        self.user_prompt_1 = """The problem your student is solving is on the topic: {topic}
        Conversation History: {history}

        Tutor response (maximum one sentence that is most appropriate given topic and conversation history)"""

        self.user_prompt = """### Task Description:
    Please respond in alignment with the conversation topic and conversation history.

    ### User:
    Conversation Topic: {topic}

    Conversation History: {history}

    ### Assistant:
    Tutor response (maximum one sentence):"""

        # self.combined_prompt = """### System:
        # You are an experienced middle school math teacher, and you are going to respond to astudent's mistake in a useful and caring way
        # ### User:
        # The problem your student is solving is on the topic: {topic}
        # Conversation History: {history}
        # ### Assistant:
        # Tutor response (maximum one sentence that is most appropriate given topic and conversation history):"""

    def gen_prompt(self, streamdata):
        # remove \u00a0 in conversation_history
        streamdata["conversation_history"] = streamdata["conversation_history"].replace(
            "\u00a0", " "
        )
        user_content = self.user_prompt.format(
            topic=streamdata["Topic"], history=streamdata["conversation_history"]
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]


class prompt_MathDial:

    def __init__(self):
        self.system_prompt = """You are an experienced middle school math teacher, and you are going to respond to astudent's mistake in a useful and caring way"""
        self.user_prompt_1 = """The conversation history of the problem your student is solving is: {history}
        
        Tutor response (maximum one sentence that is most appropriate given topic and conversation history)"""

        self.user_prompt = """### Task Description:
    Please respond in alignment with the conversation topic and conversation history.

    ### User:
    Conversation History: {history}

    ### Assistant:
    Tutor response (maximum one sentence):"""

        # self.combined_prompt = """### System:
        # You are an experienced middle school math teacher, and you are going to respond to astudent's mistake in a useful and caring way
        # ### User:
        # The conversation history of the problem your student is solving is: {history}
        # ### Assistant:
        # Tutor response (maximum one sentence that is most appropriate given topic and conversation history):"""

    def gen_prompt(self, streamdata):
        streamdata["conversation_history"] = streamdata["conversation_history"].replace(
            "\u00a0", " "
        )
        user_content = self.user_prompt.format(
            history=streamdata["conversation_history"]
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
