# LangChain, LangGraph & MCP — Explained Like You're Seven 🧸

> A friendly, picture-in-your-head guide to building smart chatbots.
> Every topic has: a **simple story**, a **plain explanation**, and **real code** with both
> **static inputs** (you type the value into the code) and **dynamic inputs** (the value comes from the user / a variable at run time).

---

## 🪄 The Big Picture (read this first)

Imagine you are building a **robot helper** out of LEGO.

- **LangChain** is the box of LEGO pieces — model "brains," tools, memory, prompts.
- **An Agent** is a LEGO robot that can *think* and *decide* "should I use a tool or just answer?"
- **LangGraph** is the **instruction booklet with a map**. It draws boxes (jobs) and arrows (what happens next), so your robot follows a clear path instead of getting lost.
- **A Node** is one box on the map = one job ("ask the brain", "use a calculator").
- **An Edge** is an arrow = "after this job, go there."
- **A Checkpointer** is a **save button in a video game** — it remembers where you were.
- **MCP** is a **universal plug (like USB‑C)** so any tool can plug into any robot.

Keep this LEGO picture in your head. Every section below is just one more LEGO idea.

### Install everything once

```bash
pip install langchain langgraph langchain-openai langchain-anthropic \
            langchain-mcp-adapters fastmcp pillow python-dotenv
```

Set your key (put this in a file called `.env`):

```
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...
```

```python
# Load the key at the top of any script
from dotenv import load_dotenv
load_dotenv()
```

---

## 1. 🤖 LangChain Agents

### The story
A **plain chatbot** is like a kid who can only *talk*. An **agent** is a kid who can *talk* **and** *go fetch things*: "Hmm, you asked for today's weather — let me go look it up, then tell you." The agent **decides by itself** whether to use a tool.

### Plain explanation
An agent = **a model (brain) + a list of tools + a loop**. The loop is:
1. Read the question.
2. Think: "Do I need a tool?"
3. If yes → call the tool, read the result, think again.
4. If no → give the final answer.

This "think → act → look → think again" loop is called **ReAct** (Reason + Act).

### Code — static input
Here the question is **hard-coded** inside the file.

```python
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent   # LangChain v1 style
from langchain_core.tools import tool

# --- a tool is just a normal Python function with a @tool sticker ---
@tool
def get_weather(city: str) -> str:
    """Return the weather for a city."""
    fake_db = {"paris": "22°C and sunny", "kanpur": "34°C and hazy"}
    return fake_db.get(city.lower(), "No data for that city")

# Build the agent: pick a brain, hand it the tools
agent = create_agent("openai:gpt-4.1", tools=[get_weather])

# STATIC input — the question is written right here
result = agent.invoke({"messages": [{"role": "user",
                                      "content": "What's the weather in Paris?"}]})
print(result["messages"][-1].content)
```

> Older but still very common: `from langgraph.prebuilt import create_react_agent` — it works almost the same way (`create_react_agent(model, tools)`). Use whichever your tutorial uses; the *idea* is identical.

### Code — dynamic input
Now the question comes from a **variable / the keyboard**, so the same agent answers anything.

```python
def ask_agent(user_question: str) -> str:
    """DYNAMIC input: question is decided at run time."""
    out = agent.invoke({"messages": [{"role": "user", "content": user_question}]})
    return out["messages"][-1].content

# the value is NOT in the code — it comes from the user
while True:
    q = input("You: ")
    if q.lower() in {"quit", "exit"}:
        break
    print("Bot:", ask_agent(q))
```

**Static vs dynamic in one line:** static = the value lives *in the code*; dynamic = the value arrives *while the program runs*.

---

## 2. ✋ Human-in-the-Loop (+ Interrupts)

### The story
Your robot wants to press the big red **"DELETE EVERYTHING"** button. Smart robots **stop and ask a grown-up first**: "Is it okay if I do this?" It freezes, waits for a thumbs-up, then continues. That pause is an **interrupt**.

### Plain explanation
- **`interrupt(value)`** is a magic word you put *inside a node*. When the program reaches it, the graph **freezes** and hands `value` back to you (the human).
- You look at it, then **resume** by calling the graph again with **`Command(resume=your_answer)`**.
- The frozen state is saved by a **checkpointer** (the save button), so it can wait seconds *or months*.
- ⚠️ **You must attach a checkpointer**, or interrupts can't work (there's nowhere to save the pause).

### Code — static input (auto-approve)
Here we pretend the human always says "yes" — the resume value is **hard-coded**.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    action: str
    status: str

def ask_human(state: State):
    # Graph FREEZES here and shows this to the human
    decision = interrupt({"question": f"Approve action: {state['action']}?"})
    # resumes here once the human answers
    return {"status": "done" if decision == "approve" else "cancelled"}

builder = StateGraph(State)
builder.add_node("ask_human", ask_human)
builder.add_edge(START, "ask_human")
builder.add_edge("ask_human", END)

graph = builder.compile(checkpointer=InMemorySaver())   # save button REQUIRED
config = {"configurable": {"thread_id": "demo-1"}}       # which save-slot

# 1) run until it freezes
first = graph.invoke({"action": "delete database", "status": ""}, config)
print(first["__interrupt__"])     # shows the question

# 2) STATIC resume value — we always approve
final = graph.invoke(Command(resume="approve"), config)
print(final["status"])            # -> done
```

### Code — dynamic input (real human decides)
Now the answer comes from a **real person** at run time.

```python
# 1) run until interrupt
state = graph.invoke({"action": "delete database", "status": ""}, config)

# 2) DYNAMIC: ask the actual human
question = state["__interrupt__"][0].value["question"]
human_says = input(f"{question} (type approve/reject): ")   # comes from keyboard

# 3) resume with whatever they typed
final = graph.invoke(Command(resume=human_says), config)
print("Result:", final["status"])
```

**Key idea:** the node *re-runs from its start* when resumed, and `interrupt()` now returns your answer instead of freezing. Same `thread_id` = it remembers where it paused.

---

## 3. 🗺️ LangGraph — Graph Creation & Types

### The story
A graph is a **treasure map**. Boxes are places to visit; arrows show the path. Some maps go in a **straight line**, some have **forks** (go left or right), and some **loop back** (try again).

### Plain explanation
You build a graph in 3 steps:
1. Define the **State** — a shared backpack 🎒 that every box can read and add to.
2. Add **nodes** (boxes / jobs) and **edges** (arrows).
3. **Compile** it into a runnable graph.

**Types of graphs you'll meet:**

| Type | Looks like | When to use |
|------|-----------|-------------|
| **Linear / sequential** | A → B → C | simple step-by-step pipeline |
| **Conditional / branching** | A → (B or C) | decide path based on data |
| **Looping / cyclic** | A → B → A … | agents that retry until done |
| **Parallel (fan-out/fan-in)** | A → (B & C) → D | do several things at once |

### Code — static input (linear graph)

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    text: str

def make_upper(state: State):
    return {"text": state["text"].upper()}

def add_excite(state: State):
    return {"text": state["text"] + "!!!"}

g = StateGraph(State)
g.add_node("upper", make_upper)
g.add_node("excite", add_excite)
g.add_edge(START, "upper")     # straight line
g.add_edge("upper", "excite")
g.add_edge("excite", END)
app = g.compile()

print(app.invoke({"text": "hello"}))   # STATIC -> {'text': 'HELLO!!!'}
```

### Code — dynamic input (conditional / branching graph)
The path is chosen **at run time** based on the user's number.

```python
class State(TypedDict):
    number: int
    label: str

def check(state: State):       # a node that just looks
    return {}

def even_node(state: State):
    return {"label": "even"}

def odd_node(state: State):
    return {"label": "odd"}

# the ROUTER decides the arrow dynamically
def router(state: State) -> str:
    return "even_node" if state["number"] % 2 == 0 else "odd_node"

g = StateGraph(State)
g.add_node("check", check)
g.add_node("even_node", even_node)
g.add_node("odd_node", odd_node)
g.add_edge(START, "check")
g.add_conditional_edges("check", router, ["even_node", "odd_node"])
g.add_edge("even_node", END)
g.add_edge("odd_node", END)
app = g.compile()

n = int(input("Give a number: "))         # DYNAMIC
print(app.invoke({"number": n, "label": ""}))
```

---

## 4. ⬛➡️ Nodes & Edges — Create, Add, Remove

### The story
Nodes are **rooms** in a house; edges are **doors** between rooms. Building the house = adding rooms and doors. Remodeling = removing a door and putting it somewhere else.

### Plain explanation
- **Node** = a function `def f(state): return {...}`. Whatever it returns gets **merged into the backpack** (state).
- **Normal edge**: `add_edge("A", "B")` → always go A then B.
- **Conditional edge**: `add_conditional_edges("A", router_fn, [...])` → A then *maybe* B or C.
- **START** and **END** are special built-in nodes (the front door and back door).

### How to ADD nodes and edges

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class S(TypedDict):
    value: int

builder = StateGraph(S)

# ADD nodes (name, function)
builder.add_node("double", lambda s: {"value": s["value"] * 2})
builder.add_node("plus_one", lambda s: {"value": s["value"] + 1})

# ADD edges
builder.add_edge(START, "double")
builder.add_edge("double", "plus_one")
builder.add_edge("plus_one", END)

graph = builder.compile()
print(graph.invoke({"value": 5}))   # {'value': 11}  (5*2 + 1)
```

### How to REMOVE a node or edge
LangGraph doesn't have a "delete this door" button after compiling. Instead, you **rebuild the graph without that piece** (think: you redraw the map). The clean pattern is a builder function:

```python
def build_graph(include_plus_one: bool = True):
    b = StateGraph(S)
    b.add_node("double", lambda s: {"value": s["value"] * 2})
    b.add_edge(START, "double")

    if include_plus_one:                       # DYNAMIC: keep the node
        b.add_node("plus_one", lambda s: {"value": s["value"] + 1})
        b.add_edge("double", "plus_one")
        b.add_edge("plus_one", END)
    else:                                       # "removed" -> just don't add it
        b.add_edge("double", END)
    return b.compile()

print(build_graph(True).invoke({"value": 5}))   # 11
print(build_graph(False).invoke({"value": 5}))  # 10  (plus_one removed)
```

**Static vs dynamic here:** `include_plus_one=True` written in code is static; passing it in from a config/user choice makes the shape of the graph dynamic.

---

## 5. 🖼️ Multimodal Workflow

### The story
"Multimodal" just means **more than one kind of input** — not only words, but also **pictures** (and sometimes audio). It's a robot with **eyes**, not just **ears**.

### Plain explanation
You send the model a **list of content pieces**: some are `text`, some are `image`. The model looks at the picture and reads the text together, then answers.

### Code — static input (image file path is fixed)

```python
import base64
from langchain_openai import ChatOpenAI   # or ChatAnthropic
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model="gpt-4.1")       # a model with "eyes"

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# STATIC: the file name and question are hard-coded
img_b64 = encode_image("cat.jpg")
message = HumanMessage(content=[
    {"type": "text", "text": "What animal is in this picture?"},
    {"type": "image_url",
     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
])
print(model.invoke([message]).content)
```

### Code — dynamic input (image + question come from the user)
Now wrap it in a tiny LangGraph so it fits the rest of the system.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class VisionState(TypedDict):
    image_path: str
    question: str
    answer: str

def look_node(state: VisionState):
    b64 = encode_image(state["image_path"])
    msg = HumanMessage(content=[
        {"type": "text", "text": state["question"]},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ])
    return {"answer": model.invoke([msg]).content}

g = StateGraph(VisionState)
g.add_node("look", look_node)
g.add_edge(START, "look")
g.add_edge("look", END)
vision_app = g.compile()

# DYNAMIC: both values supplied at run time
path = input("Image path: ")
question = input("Ask about the image: ")
result = vision_app.invoke({"image_path": path, "question": question, "answer": ""})
print(result["answer"])
```

---

## 6. 👥 Multi-Agent Workflow

### The story
One robot is good. A **team** of robots is better: a **Researcher** robot finds facts, a **Writer** robot turns them into a nice paragraph, and a **Boss** robot decides who works next. Each is good at *one* job.

### Plain explanation
A multi-agent system = several specialist agents + a way to **hand off** work between them. In LangGraph, each agent is a **node**; handoffs are **edges** (or a router that picks the next agent). A common shape is **Supervisor**: a boss node routes to the right worker.

### Code — static input (fixed pipeline: research → write)

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1")

class TeamState(TypedDict):
    topic: str
    facts: str
    article: str

def researcher(state: TeamState):
    prompt = f"List 3 quick facts about {state['topic']}."
    return {"facts": llm.invoke(prompt).content}

def writer(state: TeamState):
    prompt = f"Write a fun 3-sentence note using these facts:\n{state['facts']}"
    return {"article": llm.invoke(prompt).content}

g = StateGraph(TeamState)
g.add_node("researcher", researcher)
g.add_node("writer", writer)
g.add_edge(START, "researcher")
g.add_edge("researcher", "writer")   # handoff
g.add_edge("writer", END)
team = g.compile()

# STATIC topic
print(team.invoke({"topic": "honey bees", "facts": "", "article": ""})["article"])
```

### Code — dynamic input (Supervisor picks the worker at run time)

```python
class State(TypedDict):
    request: str
    result: str

def math_agent(state): return {"result": f"[math] solved: {state['request']}"}
def joke_agent(state): return {"result": f"[joke] here's a joke about: {state['request']}"}

def supervisor(state: State) -> str:
    """Boss reads the request and routes dynamically."""
    text = state["request"].lower()
    if any(w in text for w in ["add", "plus", "calculate", "math"]):
        return "math_agent"
    return "joke_agent"

g = StateGraph(State)
g.add_node("math_agent", math_agent)
g.add_node("joke_agent", joke_agent)
g.add_conditional_edges(START, supervisor, ["math_agent", "joke_agent"])
g.add_edge("math_agent", END)
g.add_edge("joke_agent", END)
app = g.compile()

req = input("What do you need? ")               # DYNAMIC
print(app.invoke({"request": req, "result": ""})["result"])
```

---

## 7. 🧰 Multi-Tool Workflow

### The story
A Swiss Army knife has many tools. The robot looks at the job and **picks the right tool** — scissors for paper, screwdriver for screws.

### Plain explanation
Give the model **several tools**. It chooses which to call (maybe more than one). The easiest way is a prebuilt agent; the "manual" way uses a `ToolNode` + a `tools_condition` arrow that loops back to the model until it's done.

### Code — static input (prebuilt agent, fixed question)

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def to_upper(text: str) -> str:
    """Make text uppercase."""
    return text.upper()

agent = create_agent("openai:gpt-4.1", tools=[add, to_upper])

# STATIC question that needs the 'add' tool
out = agent.invoke({"messages": [{"role": "user", "content": "what is 7 plus 5?"}]})
print(out["messages"][-1].content)
```

### Code — dynamic input (manual graph with a tool loop)
This is the classic "agent loop" you'll see everywhere.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI

tools = [add, to_upper]
model = ChatOpenAI(model="gpt-4.1").bind_tools(tools)   # let the model see the tools

def call_model(state: MessagesState):
    return {"messages": [model.invoke(state["messages"])]}

g = StateGraph(MessagesState)
g.add_node("model", call_model)
g.add_node("tools", ToolNode(tools))
g.add_edge(START, "model")
# if the model asked for a tool -> go to "tools", else -> END
g.add_conditional_edges("model", tools_condition)
g.add_edge("tools", "model")        # loop back so the model can read the result
app = g.compile()

user_q = input("Ask me to add numbers or shout text: ")    # DYNAMIC
final = app.invoke({"messages": [{"role": "user", "content": user_q}]})
print(final["messages"][-1].content)
```

---

## 8. 💾 Persistence

### The story
Without memory, your robot forgets your name the moment you turn around — like a goldfish. **Persistence** gives it a notebook so it remembers your whole conversation, even after the program closes.

### Plain explanation
Persistence = **saving the graph's state** so it survives across turns (and even restarts). LangGraph does this with a **checkpointer** + a **`thread_id`** (the name of the conversation). Same `thread_id` → it loads your old backpack and continues.

### Code — static input (in-memory, same thread)

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4.1")

def chat(state: MessagesState):
    return {"messages": [model.invoke(state["messages"])]}

g = StateGraph(MessagesState)
g.add_node("chat", chat)
g.add_edge(START, "chat")
g.add_edge("chat", END)
app = g.compile(checkpointer=InMemorySaver())   # <-- gives it memory

config = {"configurable": {"thread_id": "alice-chat"}}  # STATIC conversation name

app.invoke({"messages": [{"role": "user", "content": "My name is Alice."}]}, config)
out = app.invoke({"messages": [{"role": "user", "content": "What's my name?"}]}, config)
print(out["messages"][-1].content)   # -> remembers "Alice"
```

### Code — dynamic input (each user gets their own memory slot)

```python
def chat_turn(user_id: str, message: str) -> str:
    cfg = {"configurable": {"thread_id": user_id}}   # DYNAMIC: id per user
    out = app.invoke({"messages": [{"role": "user", "content": message}]}, cfg)
    return out["messages"][-1].content

# two different people = two separate notebooks
print(chat_turn("bob",   "I love trains."))
print(chat_turn("carol", "I love cats."))
print(chat_turn("bob",   "What do I love?"))   # -> trains (not cats!)
```

---

## 9. 🗃️ Checkpointers

### The story
A checkpointer is the **save file** in a video game. `InMemorySaver` saves to the computer's short-term memory (gone when you close the game). `SqliteSaver` saves to a real file on disk (still there tomorrow).

### Plain explanation
A checkpointer is the *thing that does the saving* for persistence. Pick one based on how long you need the memory:

| Checkpointer | Saves to | Lasts after restart? | Use for |
|--------------|----------|----------------------|---------|
| `InMemorySaver` | RAM | ❌ No | quick demos, tests |
| `SqliteSaver` | a `.db` file | ✅ Yes | small real apps |
| `AsyncSqliteSaver` | a `.db` file | ✅ Yes | async apps (FastAPI) |
| Postgres saver | a database | ✅ Yes | production / many users |

### Code — static input (SQLite file, fixed path)

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# STATIC path: always the same file
with SqliteSaver.from_conn_string("memory.db") as saver:
    app = g.compile(checkpointer=saver)
    cfg = {"configurable": {"thread_id": "alice-chat"}}
    app.invoke({"messages": [{"role": "user", "content": "Remember: sky is blue."}]}, cfg)
    print(app.invoke(
        {"messages": [{"role": "user", "content": "What color is the sky?"}]}, cfg
    )["messages"][-1].content)
# Close the program, run again -> it STILL remembers, because it's on disk.
```

### Code — dynamic input (choose the saver at run time)

```python
def make_app(durable: bool):
    if durable:                                   # DYNAMIC choice
        from langgraph.checkpoint.sqlite import SqliteSaver
        saver = SqliteSaver.from_conn_string("memory.db").__enter__()
    else:
        from langgraph.checkpoint.memory import InMemorySaver
        saver = InMemorySaver()
    return g.compile(checkpointer=saver)

use_disk = input("Save to disk? (y/n) ").lower().startswith("y")
app = make_app(use_disk)
```

> Install the SQLite saver if needed: `pip install langgraph-checkpoint-sqlite`

---

## 10. ▶️ Graph Execution

### The story
Once the map is drawn, you have to **walk it**. You can walk it fast (just tell me the final answer = `invoke`) or watch every step (`stream`), and you can do it normally or with `await` (async, good for web servers).

### Plain explanation
Three main ways to run a compiled graph:

- **`graph.invoke(input, config)`** → run to the end, return final state. Simple.
- **`graph.stream(input, config)`** → yields updates step by step (great for "typing…" UIs).
- **`graph.ainvoke / astream`** → the same but **async** (`await`), for FastAPI/servers.

`config` carries the `thread_id` (for memory) and any runtime settings.

### Code — static input (plain invoke + stream)

```python
# Final answer only:
final = app.invoke({"messages": [{"role": "user", "content": "Hi!"}]},
                   {"configurable": {"thread_id": "t1"}})
print(final["messages"][-1].content)

# Watch each step (STATIC question):
for step in app.stream({"messages": [{"role": "user", "content": "Hi!"}]},
                       {"configurable": {"thread_id": "t1"}},
                       stream_mode="updates"):
    print("STEP:", step)
```

### Code — dynamic input (async server style)

```python
import asyncio

async def run_async(user_msg: str, thread: str):
    cfg = {"configurable": {"thread_id": thread}}       # DYNAMIC thread + msg
    result = await app.ainvoke(
        {"messages": [{"role": "user", "content": user_msg}]}, cfg)
    return result["messages"][-1].content

print(asyncio.run(run_async("Tell me a fact", "user-42")))
```

> Remember: anything using **interrupts or persistence** needs `config` with a `thread_id`, every single call.

---

## 11. 🧠🧰 Multi-Agent + Multi-Tool Chatbot

### The story
Now we combine ideas 6 and 7: a **team of robots**, where **each robot also has its own tools**. A boss reads your message and sends it to the right specialist; that specialist uses its tools and answers.

### Plain explanation
- **Supervisor** node routes the message.
- Each **worker** is itself a small tool-using agent.
- They all share one backpack (`MessagesState`), with memory via a checkpointer.

### Code — static + dynamic together

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1")

# --- tools for the MATH worker ---
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# --- tools for the TRAVEL worker ---
@tool
def get_weather(city: str) -> str:
    """Weather for a city."""
    return {"paris": "22°C sunny", "kanpur": "34°C hazy"}.get(city.lower(), "unknown")
@tool
def time_zone(city: str) -> str:
    """Time zone for a city."""
    return {"paris": "CET", "kanpur": "IST"}.get(city.lower(), "unknown")

# each worker is a mini tool-using agent
math_agent   = create_react_agent(llm, [add, multiply])
travel_agent = create_react_agent(llm, [get_weather, time_zone])

def math_node(state: MessagesState):
    return math_agent.invoke(state)
def travel_node(state: MessagesState):
    return travel_agent.invoke(state)

# Supervisor routes DYNAMICALLY by reading the last user message
def supervisor(state: MessagesState) -> str:
    text = state["messages"][-1].content.lower()
    if any(w in text for w in ["add", "times", "multiply", "plus", "calculate"]):
        return "math"
    if any(w in text for w in ["weather", "time", "city", "travel"]):
        return "travel"
    return "math"   # default

builder = StateGraph(MessagesState)
builder.add_node("math", math_node)
builder.add_node("travel", travel_node)
builder.add_conditional_edges(START, supervisor, ["math", "travel"])
builder.add_edge("math", END)
builder.add_edge("travel", END)
bot = builder.compile(checkpointer=InMemorySaver())

def chat(message: str, user="u1"):
    cfg = {"configurable": {"thread_id": user}}
    out = bot.invoke({"messages": [{"role": "user", "content": message}]}, cfg)
    return out["messages"][-1].content

# STATIC demo
print(chat("what is 6 times 7?"))
# DYNAMIC loop
# while True: print("Bot:", chat(input("You: ")))
```

---

## 12. 🔗 Workflow + Agents (when to use which)

### The story
A **workflow** is a train on **fixed tracks** — predictable, every time the same stops. An **agent** is a taxi — it **decides the route** based on traffic. Real apps mix both: a fixed train line, but one stop is "let the taxi figure it out."

### Plain explanation
- **Workflow** = *you* hard-code the steps and order. Reliable, easy to debug. Best when the process is known.
- **Agent** = *the model* decides steps and tool order. Flexible, handles surprises. Best when the path is unknown.
- **Mix**: put an agent **inside one node** of a larger fixed workflow. You get reliability *and* flexibility.

### Code — a fixed workflow with an agent node in the middle

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

llm = ChatOpenAI(model="gpt-4.1")

@tool
def word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())

inner_agent = create_react_agent(llm, [word_count])

class State(TypedDict):
    raw: str          # fixed step 1 output
    smart: str        # agent step output
    final: str        # fixed step 3 output

def clean_step(state: State):                 # FIXED workflow step
    return {"raw": state["raw"].strip().lower()}

def agent_step(state: State):                 # AGENT step (flexible)
    out = inner_agent.invoke(
        {"messages": [{"role": "user",
                       "content": f"How many words is this? '{state['raw']}'"}]})
    return {"smart": out["messages"][-1].content}

def format_step(state: State):                # FIXED workflow step
    return {"final": f"RESULT >> {state['smart']}"}

g = StateGraph(State)
g.add_node("clean", clean_step)
g.add_node("agent", agent_step)
g.add_node("format", format_step)
g.add_edge(START, "clean")
g.add_edge("clean", "agent")
g.add_edge("agent", "format")
g.add_edge("format", END)
app = g.compile()

# STATIC
print(app.invoke({"raw": "  Hello Brave New World  ", "smart": "", "final": ""})["final"])
# DYNAMIC: feed app.invoke({"raw": input("Text: "), ...})
```

---

## 13. 🌐 API Integration Guide

### The story
You built the robot. Now you put it behind a **front desk window** so anyone on the internet can pass a note through the slot and get an answer back. That window is an **API** (we'll use **FastAPI**).

### Plain explanation
Wrap your compiled graph in a web server. The user sends **JSON** (`{"message": "...", "user_id": "..."}`); you run the graph; you send the answer back as JSON. Two endpoints are handy: a normal one (`/chat`) and a streaming one (`/chat/stream`).

### Code — FastAPI server (dynamic input is the whole point of an API)

```python
# file: server.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.memory import InMemorySaver
# (import your graph builder `g` from earlier, e.g. the multi-tool one)

app_api = FastAPI(title="My Chatbot API")
graph = g.compile(checkpointer=InMemorySaver())

class ChatRequest(BaseModel):     # defines the JSON the user must send
    message: str
    user_id: str = "default"      # has a STATIC default if not given

class ChatResponse(BaseModel):
    reply: str

@app_api.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):                 # DYNAMIC: values from the request
    cfg = {"configurable": {"thread_id": req.user_id}}
    out = await graph.ainvoke(
        {"messages": [{"role": "user", "content": req.message}]}, cfg)
    return ChatResponse(reply=out["messages"][-1].content)

@app_api.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    cfg = {"configurable": {"thread_id": req.user_id}}
    async def gen():
        async for chunk in graph.astream(
            {"messages": [{"role": "user", "content": req.message}]},
            cfg, stream_mode="messages"):
            token = chunk[0].content if isinstance(chunk, tuple) else ""
            if token:
                yield token
    return StreamingResponse(gen(), media_type="text/plain")

@app_api.get("/health")
def health():
    return {"status": "ok"}
```

Run it and call it:

```bash
uvicorn server:app_api --reload --port 8080
```

```bash
# STATIC test from the terminal
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "what is 2 plus 2?", "user_id": "ravi"}'
```

```python
# DYNAMIC test from Python
import requests
msg = input("You: ")
r = requests.post("http://localhost:8080/chat",
                  json={"message": msg, "user_id": "ravi"})
print("Bot:", r.json()["reply"])
```

---

## 14. 🔌 MCP with FastMCP (running, types, using in a chatbot API)

### The story
Every tool used to need its own special cable. **MCP (Model Context Protocol)** is **USB-C for AI tools**: build a tool once, and *any* MCP-aware robot can plug into it. **FastMCP** is the easiest way to build such a tool server in Python.

### Plain explanation — the 3 things an MCP server can offer
| Primitive | Like a… | What it does |
|-----------|---------|--------------|
| **Tool** (`@mcp.tool`) | POST button | *does* something / runs code |
| **Resource** (`@mcp.resource`) | GET page | *gives* read-only data/info |
| **Prompt** (`@mcp.prompt`) | recipe card | a reusable prompt template |

**Transports = how clients connect:**
- **stdio** (default): runs as a local child process. Best for local/desktop use.
- **http** (streamable): a real web URL. Best for remote / many clients.
- **sse**: older, kept only for backward compatibility — avoid for new projects.

### Step 1 — Build & run an MCP server (FastMCP)

```python
# file: math_server.py
from fastmcp import FastMCP

mcp = FastMCP("MathServer")

@mcp.tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@mcp.tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@mcp.resource("resource://greeting")
def greeting() -> str:
    """A friendly read-only message."""
    return "Hello from the MCP math server!"

@mcp.prompt
def explain_prompt(topic: str) -> str:
    """Reusable prompt template."""
    return f"Explain {topic} to a 7-year-old in 3 short sentences."

if __name__ == "__main__":
    # STATIC: stdio is the default (local)
    mcp.run()
    # DYNAMIC choice -> run as a web service instead:
    # mcp.run(transport="http", host="127.0.0.1", port=8000)
```

Run locally:

```bash
python math_server.py                       # stdio (default)
# or as a web server:
fastmcp run math_server.py --transport http --port 8000
```

### Step 2 — Use the MCP server inside a chatbot (LangChain adapter)
This connects your robot's brain to the MCP tools. Note MCP is **async**, so we use `await`.

```python
# file: mcp_chatbot.py
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

async def main():
    # connect to one OR many MCP servers (static config here)
    client = MultiServerMCPClient({
        "math": {                       # local server via stdio
            "transport": "stdio",
            "command": "python",
            "args": ["math_server.py"], # use a full absolute path in real apps
        },
        # add an HTTP one too if you ran it on a port:
        # "weather": {"transport": "http", "url": "http://localhost:8000/mcp"},
    })

    tools = await client.get_tools()           # MCP tools -> LangChain tools
    agent = create_react_agent(ChatOpenAI(model="gpt-4.1"), tools)

    async def ask(q: str):                     # DYNAMIC question
        out = await agent.ainvoke({"messages": [{"role": "user", "content": q}]})
        return out["messages"][-1].content

    print(await ask("what is (3 + 5) times 12?"))   # uses add + multiply tools

asyncio.run(main())
```

### Step 3 — Put the MCP-powered bot behind an API
Combine §13 and §14: load MCP tools once at startup, serve them over HTTP.

```python
# file: mcp_api.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

STATE = {}   # holds the built agent

@asynccontextmanager
async def lifespan(app: FastAPI):
    client = MultiServerMCPClient({
        "math": {"transport": "stdio", "command": "python", "args": ["math_server.py"]},
    })
    tools = await client.get_tools()
    STATE["agent"] = create_react_agent(
        ChatOpenAI(model="gpt-4.1"), tools, checkpointer=InMemorySaver())
    yield
    STATE.clear()

api = FastAPI(lifespan=lifespan)

class Req(BaseModel):
    message: str
    user_id: str = "default"

@api.post("/chat")
async def chat(req: Req):                       # DYNAMIC input from the web
    cfg = {"configurable": {"thread_id": req.user_id}}
    out = await STATE["agent"].ainvoke(
        {"messages": [{"role": "user", "content": req.message}]}, cfg)
    return {"reply": out["messages"][-1].content}
```

```bash
uvicorn mcp_api:api --reload --port 8080
```

---

## 15. 🏗️ Mini Project — Multimodal + Multi-Agent + Multi-Tool Chatbot

A small but complete app that ties **everything** together:
- 👁️ **Multimodal**: can look at images.
- 👥 **Multi-agent**: a Supervisor routes to a **Vision agent**, a **Math agent**, or a **Travel agent**.
- 🧰 **Multi-tool**: each agent has its own tools (some real Python tools, some via an **MCP server**).
- 💾 **Persistence**: remembers each user with a SQLite checkpointer.
- 🌐 **API**: served with FastAPI.

### 📁 Project structure

```
smart-chatbot/
├── .env                     # API keys (never commit this)
├── requirements.txt
├── mcp_servers/
│   └── math_server.py       # FastMCP server: add, multiply (stdio)
├── app/
│   ├── __init__.py
│   ├── tools.py             # local tools (weather, time_zone, word_count)
│   ├── agents.py            # vision / math / travel agents
│   ├── graph.py             # supervisor graph + checkpointer
│   └── server.py            # FastAPI app (/chat, /chat/image, /health)
└── README.md
```

### 📦 `requirements.txt`

```
langchain>=1.0
langgraph>=1.0
langchain-openai>=0.2
langchain-mcp-adapters>=0.1
langgraph-checkpoint-sqlite>=2.0
fastmcp>=2.0
fastapi>=0.115
uvicorn[standard]>=0.30
pydantic>=2.0
python-dotenv>=1.0
pillow>=10.0
python-multipart>=0.0.9
```

Install: `pip install -r requirements.txt`

### `mcp_servers/math_server.py`

```python
from fastmcp import FastMCP

mcp = FastMCP("MathServer")

@mcp.tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@mcp.tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

if __name__ == "__main__":
    mcp.run()   # stdio
```

### `app/tools.py`

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    db = {"paris": "22°C sunny", "kanpur": "34°C hazy", "tokyo": "18°C rainy"}
    return db.get(city.lower(), "No data for that city")

@tool
def time_zone(city: str) -> str:
    """Get the time zone of a city."""
    db = {"paris": "CET", "kanpur": "IST", "tokyo": "JST"}
    return db.get(city.lower(), "Unknown time zone")

@tool
def word_count(text: str) -> int:
    """Count the number of words in some text."""
    return len(text.split())
```

### `app/agents.py`

```python
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from .tools import get_weather, time_zone, word_count

llm = ChatOpenAI(model="gpt-4.1")

# travel agent uses LOCAL tools
travel_agent = create_react_agent(llm, [get_weather, time_zone, word_count])

# (math tools come from MCP; wired in graph.py)

def make_math_agent(mcp_tools):
    return create_react_agent(llm, mcp_tools)

# Vision is not a tool-agent; it's a direct multimodal call
def vision_answer(image_b64: str, question: str) -> str:
    msg = HumanMessage(content=[
        {"type": "text", "text": question},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
    ])
    return llm.invoke([msg]).content
```

### `app/graph.py`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from .agents import travel_agent, make_math_agent

async def build_bot():
    # 1) connect to the MCP math server (multi-tool via MCP)
    client = MultiServerMCPClient({
        "math": {"transport": "stdio", "command": "python",
                 "args": ["mcp_servers/math_server.py"]},
    })
    math_tools = await client.get_tools()
    math_agent = make_math_agent(math_tools)

    # 2) worker nodes
    def math_node(state: MessagesState):   return math_agent.invoke(state)
    def travel_node(state: MessagesState): return travel_agent.invoke(state)

    # 3) supervisor routes DYNAMICALLY
    def supervisor(state: MessagesState) -> str:
        text = state["messages"][-1].content.lower()
        if any(w in text for w in ["add", "plus", "times", "multiply", "calculate"]):
            return "math"
        return "travel"

    # 4) assemble graph with persistence
    saver = SqliteSaver.from_conn_string("chatbot.db").__enter__()
    g = StateGraph(MessagesState)
    g.add_node("math", math_node)
    g.add_node("travel", travel_node)
    g.add_conditional_edges(START, supervisor, ["math", "travel"])
    g.add_edge("math", END)
    g.add_edge("travel", END)
    return g.compile(checkpointer=saver)
```

### `app/server.py`

```python
import base64
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from dotenv import load_dotenv
from .graph import build_bot
from .agents import vision_answer

load_dotenv()
STATE = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["bot"] = await build_bot()     # build once at startup
    yield
    STATE.clear()

api = FastAPI(title="Smart Chatbot", lifespan=lifespan)

class ChatReq(BaseModel):
    message: str
    user_id: str = "default"

@api.post("/chat")                       # text -> math or travel agent
async def chat(req: ChatReq):
    cfg = {"configurable": {"thread_id": req.user_id}}
    out = await STATE["bot"].ainvoke(
        {"messages": [{"role": "user", "content": req.message}]}, cfg)
    return {"reply": out["messages"][-1].content}

@api.post("/chat/image")                 # image + question -> vision agent
async def chat_image(question: str = Form(...), file: UploadFile = File(...)):
    img_b64 = base64.b64encode(await file.read()).decode("utf-8")
    return {"reply": vision_answer(img_b64, question)}

@api.get("/health")
def health():
    return {"status": "ok"}
```

### ▶️ Run it

```bash
# 1) put your key in .env
echo "OPENAI_API_KEY=sk-..." > .env

# 2) start the API (the MCP server is launched automatically by the graph)
uvicorn app.server:api --reload --port 8080
```

### 🧪 Try it

```bash
# text (routes to the MCP math agent)
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"what is 8 times 9?","user_id":"ravi"}'

# text (routes to the travel agent)
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"weather in Tokyo?","user_id":"ravi"}'

# image (vision agent)
curl -X POST http://localhost:8080/chat/image \
  -F "question=What is in this picture?" \
  -F "file=@cat.jpg"
```

### How the pieces map to the lessons
| Lesson | Where it shows up in the project |
|--------|----------------------------------|
| Agents (§1) | `create_react_agent` for math & travel |
| Human-in-loop (§2) | add `interrupt()` before a "send money" tool if you add one |
| Graph types (§3) | conditional supervisor graph |
| Nodes/edges (§4) | `add_node` / `add_conditional_edges` in `graph.py` |
| Multimodal (§5) | `/chat/image` + `vision_answer` |
| Multi-agent (§6) | supervisor → math/travel |
| Multi-tool (§7) | each agent holds several tools |
| Persistence (§8) | `thread_id = user_id` |
| Checkpointer (§9) | `SqliteSaver` → `chatbot.db` |
| Execution (§10) | `await bot.ainvoke(...)` |
| API (§13) | FastAPI endpoints |
| MCP (§14) | `math_server.py` + `MultiServerMCPClient` |

---

## 🎒 One-Page Cheat Sheet

```python
# STATE = the shared backpack
class State(TypedDict): ...

# BUILD
g = StateGraph(State)
g.add_node("name", fn)                       # add a job
g.add_edge("a", "b")                         # always a -> b
g.add_conditional_edges("a", router, [...])  # a -> maybe b/c
g.add_edge(START, "first"); g.add_edge("last", END)

# MEMORY
app = g.compile(checkpointer=InMemorySaver())     # or SqliteSaver
cfg = {"configurable": {"thread_id": "user-1"}}   # one notebook per user

# RUN
app.invoke(inp, cfg)          # final answer
app.stream(inp, cfg)          # step by step
await app.ainvoke(inp, cfg)   # async (servers)

# PAUSE FOR A HUMAN
val = interrupt({...})                 # inside a node -> freezes
app.invoke(Command(resume=answer), cfg)   # continue

# AGENT + TOOLS
agent = create_react_agent(model, [tool1, tool2])

# MCP
client = MultiServerMCPClient({"x": {"transport": "stdio",
                                     "command": "python", "args": ["s.py"]}})
tools = await client.get_tools()
```

### 🧠 Remember the 5 magic words
1. **State** = the backpack everyone shares.
2. **Node** = a job. **Edge** = an arrow to the next job.
3. **Checkpointer + thread_id** = memory.
4. **interrupt + Command(resume=…)** = ask a human, then continue.
5. **MCP** = USB-C; build a tool once, plug it in anywhere.

---

### ⚠️ Quick notes on versions
These libraries change fast. Two valid styles you'll see for building an agent:
- `from langchain.agents import create_agent` (newer LangChain v1)
- `from langgraph.prebuilt import create_react_agent` (very common, still works)

They do the same job. If an import fails, check the package's current docs and your installed version with `pip show langgraph langchain langchain-mcp-adapters fastmcp`.