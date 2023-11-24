from typing import Any

import openai
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import QueryType

from approaches.approach import ChatApproach
from core.messagebuilder import MessageBuilder
from core.modelhelper import get_token_limit
from text import nonewlines

class ChatReadRetrieveReadApproach(ChatApproach):
    # Chat roles
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    """
    Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
    top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
    (answer) with that prompt.

    Cognitive SearchとOpenAIのAPIを直接使用した、シンプルな retrieve-then-read の実装です。これは、最初に
    検索からトップ文書を抽出し、それを使ってプロンプトを構成し、OpenAIで補完生成する (answer)をそのプロンプトで表示します。
    """
    system_message_chat_conversation = """
Answer the reading comprehension question on the Labor regulations document of the Ministry of Health, Labour and Welfare in Japan.
If you cannot guess the answer to a question from the SOURCES, answer "I don't know".
Answers must be in Japanese.

# Restrictions
- The SOURCES prefix has a colon and actual information after the filename, and each fact used in the response must include the name of the source.
- To reference a source, use a square bracket. For example, [info1.txt]. Do not combine sources, but list each source separately. For example, [info1.txt][info2.pdf].

{follow_up_questions_prompt}
{injected_prompt}
"""
    follow_up_questions_prompt_content = """
Answers must be accompanied by three additional follow-up questions to the user's question. The rules for follow-up questions are defined in the Restrictions.

- Please answer only questions related to the Labor regulations document of the Ministry of Health, Labour and Welfare in Japan. If the question is not related to the Labor regulations document of the Ministry of Health, Labour and Welfare in Japan, answer "I don't know".
- Use double angle brackets to reference the questions, e.g. <<Can I do a side job? >>.
- Try not to repeat questions that have already been asked.
- Do not add SOURCES to follow-up questions.
- Do not use bulleted follow-up questions. Always enclose them in double angle brackets.
- Follow-up questions should be ideas that expand the user's curiosity.
- Only generate questions and do not generate any text before or after the questions, such as 'Next Questions'

EXAMPLE:###
Q:試用期間について教えてください？
A:試用期間を設ける場合にその期間の長さに関する定めは労基法上ありません、ただし、労働者の地位を不安定にすることから、あまりに長い期間を試用期間とすることは好ましくありません。[info1.txt][info2.pdf] 試用期間中の解雇については、最初の１４日間以内であれば即時に解雇することができます。[info3.pdf] １４日を超えて雇用した後に解雇する場合には、原則として３０日以上前に予告をしなければなりません。[info4.pdf]<<試用期間中の給与についての規則はありますか？>><<試用期間中の解雇理由はどのようになっていますか？>><<他にも試用期間に関する注意点はありますか？>>

Q:副業は出来ますか？
A:労働者は、勤務時間外において、他の会社等の業務に従事することができます。[info1.txt] ただし、これは副業・兼業に関するモデル規定であり、就業規則の内容は事業場の実態に合ったものとしなければなりません。[info2.pdf] 副業・兼業に係る相談、自己申告等を行ったことにより不利益な取扱いをすることはできません。また、労働時間の管理や労働条件の変更についても適切な措置を講じる必要があります。[info3.pdf]<<一般的な制限事項はありますか？>><<労働時間の管理はどのようになりますか？>><<他にも副業における注意点はありますか？>>
###

"""

    query_prompt_template = """
Below is a history of previous conversations and a new question from a user that needs to be answered by searching the Knowledge Base on Japanese Labor regulations document.
Based on the conversation and the new question, create a search query.
Do not include the name of the cited file or document (e.g., info.txt or doc.pdf) in the search query.
Do not include text in [] or <>> in the search query.
If you cannot generate a search query, return only the number 0.
"""
    query_prompt_few_shots = [
        {'role' : USER, 'content' : '試用期間中の解雇について教えてください  ' },
        {'role' : ASSISTANT, 'content' : '試用期間 解雇 条件' },
        {'role' : USER, 'content' : '副業の時間的制約はありますか' },
        {'role' : ASSISTANT, 'content' : '副業 時間 制約' }
    ]

    def __init__(self, search_client: SearchClient, chatgpt_deployment: str, chatgpt_model: str, embedding_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.chatgpt_model = chatgpt_model
        self.embedding_deployment = embedding_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.chatgpt_token_limit = get_token_limit(chatgpt_model)

    async def run(self, history: list[dict[str, str]], overrides: dict[str, Any]) -> Any:
        has_text = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        has_vector = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_captions = True if overrides.get("semantic_captions") and has_text else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        user_q = 'Generate search query for: ' + history[-1]["user"]

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        # チャット履歴と最後の質問に基づいて、最適化されたキーワード検索クエリを生成します。
        messages = self.get_messages_from_history(
            self.query_prompt_template,
            self.chatgpt_model,
            history,
            user_q,
            self.query_prompt_few_shots,
            self.chatgpt_token_limit - len(user_q)
            )

        chat_completion = await openai.ChatCompletion.acreate(
            deployment_id=self.chatgpt_deployment,
            model=self.chatgpt_model,
            messages=messages,
            temperature=0.0,
            max_tokens=100,
            n=1)

        query_text = chat_completion.choices[0].message.content
        print(query_text)
        if query_text.strip() == "0":
            # Use the last user input if we failed to generate a better query
            # より良いクエリを生成できなかった場合は、最後に入力されたクエリを使用する。
            query_text = history[-1]["user"] 

        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query
        # GPTで最適化されたクエリを使用して、検索インデックスから関連するドキュメントを取得します。

        # If retrieval mode includes vectors, compute an embedding for the query
        # 検索モードにベクトルが含まれている場合は、クエリの埋め込みを計算します。
        if has_vector:
            # ユーザーの入力をそのままベクトル化するアプローチも無くはない
            # query_text = history[-1]["user"] 

            query_vector = (await openai.Embedding.acreate(engine=self.embedding_deployment, input=query_text))["data"][0]["embedding"]
        else:
            query_vector = None

        # Only keep the text query if the retrieval mode uses text, otherwise drop it
        # 検索モードがテキストを使用する場合は、テキストクエリのみを保持し、それ以外は削除します。
        if not has_text:
            query_text = None

        # Use semantic L2 reranker if requested and if retrieval mode is text or hybrid (vectors + text)
        # 検索モードがテキストまたはハイブリッド（ベクトル＋テキスト）の場合、リクエストに応じてセマンティックL2リランカーを使用する。
        if overrides.get("semantic_ranker") and has_text:
            r = await self.search_client.search(query_text,
                                          filter=filter,
                                          query_type=QueryType.SEMANTIC,
                                          query_language="ja-jp", # 日本語の場合は ja-jp
                                          query_speller="none",
                                          semantic_configuration_name="default",
                                          top=top,
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None,
                                          vector=query_vector,
                                          top_k=50 if query_vector else None,
                                          vector_fields="embedding" if query_vector else None)
        else:
            r = await self.search_client.search(query_text,
                                          filter=filter,
                                          top=top,
                                          vector=query_vector,
                                          top_k=50 if query_vector else None,
                                          vector_fields="embedding" if query_vector else None)
        if use_semantic_captions:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) async for doc in r]
        else:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) async for doc in r]
        content = "\n".join(results)

        follow_up_questions_prompt = self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else ""

        # STEP 3: Generate a contextual and content specific answer using the search results and chat history
        # 検索結果とチャット履歴を使用して、文脈や内容に応じた回答を生成します。

        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        # クライアントがプロンプト全体を置き換えたり、>>を使用して終了するプロンプトに注入したりできるようにする。
        prompt_override = overrides.get("prompt_override")
        if prompt_override is None:
            system_message = self.system_message_chat_conversation.format(injected_prompt="", follow_up_questions_prompt=follow_up_questions_prompt)
        elif prompt_override.startswith(">>>"):
            system_message = self.system_message_chat_conversation.format(injected_prompt=prompt_override[3:] + "\n", follow_up_questions_prompt=follow_up_questions_prompt)
        else:
            system_message = prompt_override.format(follow_up_questions_prompt=follow_up_questions_prompt)

        messages = self.get_messages_from_history(
            system_message,
            self.chatgpt_model,
            history,
            history[-1]["user"]+ "\n\nSources:\n" + content, # Model does not handle lengthy system messages well. Moving sources to latest user conversation to solve follow up questions prompt. モデルは長いシステムメッセージをうまく扱えない。フォローアップ質問のプロンプトを解決するために、最新のユーザー会話にソースを移動する。
            max_tokens=self.chatgpt_token_limit)

        chat_completion = await openai.ChatCompletion.acreate(
            deployment_id=self.chatgpt_deployment,
            model=self.chatgpt_model,
            messages=messages,
            temperature=overrides.get("temperature") or 0.0,
            max_tokens=2048,
            n=1)

        chat_content = chat_completion.choices[0].message.content
        print(chat_content)
        msg_to_display = '\n\n'.join([str(message) for message in messages])

        return {"data_points": results, "answer": chat_content, "thoughts": f"Searched for:<br>{query_text}<br><br>Conversations:<br>" + msg_to_display.replace('\n', '<br>')}

    def get_messages_from_history(self, system_prompt: str, model_id: str, history: list[dict[str, str]], user_conv: str, few_shots = [], max_tokens: int = 4096) -> list:
        message_builder = MessageBuilder(system_prompt, model_id)

        # Add examples to show the chat what responses we want. It will try to mimic any responses and make sure they match the rules laid out in the system message.
        # どのような応答が欲しいかをチャットに示す例を追加してください。チャットはどのような応答でも模倣しようとし、システムメッセージに示されたルールに一致することを確認します。
        for shot in few_shots:
            message_builder.append_message(shot.get('role'), shot.get('content'))

        user_content = user_conv
        append_index = len(few_shots) + 1

        message_builder.append_message(self.USER, user_content, index=append_index)

        for h in reversed(history[:-1]):
            if bot_msg := h.get("bot"):
                message_builder.append_message(self.ASSISTANT, bot_msg, index=append_index)
            if user_msg := h.get("user"):
                message_builder.append_message(self.USER, user_msg, index=append_index)
            if message_builder.token_length > max_tokens:
                break

        messages = message_builder.messages
        return messages
