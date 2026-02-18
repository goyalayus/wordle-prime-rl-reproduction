# Inference vs Training Prompts

## 1. Were the prompts the same?

**No.** The inference system prompt and user prompt format are different from the training dataset.

---

## Inference System Prompt

```
You are an expert AI playing Wordle.
GOAL: Guess the secret 5-letter word in 6 tries.

FEEDBACK MEANING (per letter):
- G (Green): correct letter in the correct position.
- Y (Yellow): letter is in the word but in the wrong position.
- X (Gray): letter is not in the word.

OUTPUT FORMAT (MUST MATCH EXACTLY):
<guess>[word]</guess>

[word] must be exactly 5 lowercase letters a-z.
```

---

## Training System Prompt

```
You are a competitive game player. Make sure you read the game instructions 
carefully, and always follow the required format.

In each turn, think step-by-step inside <<think>>...<<</think> tags, then follow 
the instructions inside <<guess>>...<<</think> tags.
```

Note: The training dataset uses double-angle brackets: `<<think>>` and `<<guess>>`, not single-angle `<think>` and `<guess>`.

---

## Differences Summary

| Aspect | Training | Inference |
|--------|----------|-----------|
| Role | "competitive game player" | "expert AI playing Wordle" |
| Think tags | `<<think>>` and `<<</think>` | Uses Qwen `<think>` and `</think>` |
| Guess tags | `<<guess>>` and `<<</think>` | `<guess>` and `</guess>` |
| Feedback definition | In user message | In system prompt |

---

## 2. Simple Model Output from Training Dataset

### User message (first turn)

```
You are Player 0 in Wordle.
A secret 5-letter word has been chosen. You have 6 attempts to guess it.
For each guess, wrap your word in square brackets (e.g., [apple]).
Feedback for each letter will be given as follows:
 - G (green): correct letter in the correct position
 - Y (yellow): letter exists in the word but in the wrong position
 - X (wrong): letter is not in the word
Enter your guess to begin.
```

### Model output (first guess)

```
<<think>>
I'll start with a common 5-letter word that has a good mix of vowels and consonants 
to gather initial feedback. A good starting word is "CRANE" because it includes 
common vowels (A, E) and common consonants (C, R, N).
<<</think>>>
<<guess>>[crane]<<</think>>>
```

### User message (feedback)

```
C R A N E
X X G G G

You have 5 guesses left.
```

### Model output (second guess)

```
<<think>>
The feedback for "[crane]" is: C (X), R (X), A (G), N (G), E (G). This means:
- A, N, E are correct and in the correct positions (positions 3, 4, 5).
- C and R are not in the word at all.

So the word structure is: _ _ A N E.

Now, I need to find a 5-letter word that fits _ _ A N E, excluding C and R.
I'll try "PLANE" — P and L are new letters, A N E stay in place.
<<</think>>>
<<guess>>[plane]<<</think>>>
```

---

## Bottom Line

- **Training format:** `<<guess>>[word]<<</think>>>` with the word in square brackets
- **Inference format:** `<guess>[word]</guess>`
- **Conclusion:** There is a train–test mismatch. Aligning inference prompts and tags with the training dataset may improve performance.
