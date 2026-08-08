from typing import Annotated, TypedDict

from chains import generate_chain, reflect_chain
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph, add_messages


class MessageState(TypedDict):
    messages: Annotated[BaseMessage, add_messages]


REFLECT = "reflect"
GENERATE = "generate"
LAST = -1


def generation_node(state: MessageState):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def relection_node(state: MessageState):
    res = reflect_chain.invoke({"messages": state["messages"]})
    print(res)
    return {
        "messages": [
            HumanMessage(content={res.content[LAST]["text"] if res.content else ""})
        ]
    }


def should_continue(state: MessageState):
    if len(state["messages"]) > 6:
        return END
    return REFLECT


graph = StateGraph(state_schema=MessageState)

graph.add_node(GENERATE, generation_node)
graph.add_node(REFLECT, relection_node)
graph.set_entry_point(GENERATE)
graph.add_conditional_edges(GENERATE, should_continue, {END: END, REFLECT: REFLECT})

graph.add_edge(REFLECT, GENERATE)

app = graph.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

if __name__ == "__main__":
    print("Hello from reflection-agent!")
    inputs = {"messages": [HumanMessage(content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """)]}

    res = app.invoke(inputs)
    print(res["messages"])
