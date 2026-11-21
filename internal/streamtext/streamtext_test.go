package streamtext

import (
	"testing"

	"github.com/Samuelk0nrad/gai/ai"
)

func text(s string) ai.Token { return ai.Token{Part: &ai.ContentPart{Kind: ai.ContentText, Text: s}} }

func TestRunFollowsAppendTokenMerging(t *testing.T) {
	var run Run
	message := ai.Message{Role: ai.RoleAssistant}
	emit := func(token ai.Token) {
		run.Observe(token)
		message.AppendToken(token)
	}
	emit(text("Hëllo "))
	emit(text("wörld"))
	token, ok := run.AttachCitations("p", []ai.Citation{{URL: "u", Start: 0, End: 5}}, 5)
	if !ok {
		t.Fatal("no citation token")
	}
	emit(token)
	emit(text("next"))
	token, _ = run.AttachCitations("p", []ai.Citation{{URL: "v", Start: 0, End: 4}}, 4)
	emit(token)
	emit(ai.Token{Part: &ai.ContentPart{Kind: ai.ContentReasoning, Text: "r"}})
	if run.Len() != 0 {
		t.Fatalf("run after reasoning = %d", run.Len())
	}
	if len(message.Parts) != 3 {
		t.Fatalf("parts = %#v", message.Parts)
	}
	for i, want := range []string{"wörld", "next"} {
		part := message.Parts[i]
		c := part.Citations()[0]
		if got := SliceRunes(part.Text, c.Start, c.End); got != want {
			t.Fatalf("part %d citation covers %q, want %q", i, got, want)
		}
	}
}

func TestSliceRunes(t *testing.T) {
	for _, test := range []struct {
		s          string
		start, end int
		want       string
	}{{"héllo", 1, 3, "él"}, {"héllo", 0, 5, "héllo"}, {"héllo", 3, 9, ""}, {"héllo", 2, 2, ""}, {"", 0, 1, ""}} {
		if got := SliceRunes(test.s, test.start, test.end); got != test.want {
			t.Fatalf("SliceRunes(%q, %d, %d) = %q, want %q", test.s, test.start, test.end, got, test.want)
		}
	}
}
