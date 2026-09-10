from langgraph.graph import MessagesState


# agent state is the information carried between graph nodes. MessagesState proves a messages collection with a reducer
# we add case_id because messages describe the conversation, while case ID identifies the business execution
class AgentState(MessagesState):
    case_id: str
