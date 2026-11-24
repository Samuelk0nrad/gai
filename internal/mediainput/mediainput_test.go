package mediainput

import (
	"errors"
	"strings"
	"testing"

	"github.com/Samuelk0nrad/gai/ai"
)

func TestResolveAndCheckTextRejectWhatMappingCannotSend(t *testing.T) {
	for name, media := range map[string]ai.MediaPart{
		"bad base64":   {MIMEType: "image/png", URI: "data:image/png;base64,***"},
		"empty base64": {MIMEType: "image/png", URI: "data:image/png;base64,"},
		// The stream decoder checks padding per 1,024-byte chunk.
		"data after padding in a later chunk": {MIMEType: "image/png", URI: "data:image/png;base64," + strings.Repeat("A", 1022) + "==AAAA"},
	} {
		if _, err := Resolve("test", &media); !errors.Is(err, ai.ErrUnsupportedCapability) {
			t.Errorf("%s: err = %v", name, err)
		}
	}
	for name, media := range map[string]ai.MediaPart{
		"UTF-16 bytes": {MIMEType: "text/csv", Data: []byte{0xff, 0xfe, 'a', 0}},
		"text by URL":  {MIMEType: "text/plain", URI: "https://example.com/a.txt"},
	} {
		source, err := Resolve("test", &media)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if err := source.CheckText("test"); !errors.Is(err, ai.ErrUnsupportedCapability) {
			t.Errorf("%s: CheckText = %v", name, err)
		}
	}
}

func TestResolveAcceptsWhatBytesDecodes(t *testing.T) {
	for name, payload := range map[string]string{
		"padded":   strings.Repeat("A", 1022) + "==",
		"unpadded": strings.Repeat("A", 1024),
	} {
		source, err := Resolve("test", &ai.MediaPart{MIMEType: "image/png", URI: "data:image/png;base64," + payload})
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if _, err := source.Bytes("test"); err != nil {
			t.Fatalf("%s: Bytes = %v", name, err)
		}
	}
}
