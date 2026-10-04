package main

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/lace-ai/gai/ai"
)

// TestLookupOrderToolReturnsKnownOrder verifies that case-insensitive order lookup returns the
// canonical identifier and expected order status.
func TestLookupOrderToolReturnsKnownOrder(t *testing.T) {
	t.Parallel()

	response, callErr := newLookupOrderTool().Function(context.Background(), ai.ToolCall{
		ID:   "call_test",
		Type: "function",
		Name: "lookup_order",
		Args: json.RawMessage(`{"order_id":"lace-1042"}`),
	})
	if err := callErr; err != nil {
		t.Fatalf("lookup order: %v", err)
	}

	var result orderLookupResult
	if err := json.Unmarshal([]byte(response), &result); err != nil {
		t.Fatalf("decode result: %v", err)
	}
	if !result.Found {
		t.Fatal("expected order to be found")
	}
	if result.OrderID != "LACE-1042" {
		t.Fatalf("unexpected order ID %q", result.OrderID)
	}
	if result.Status != "in_transit" {
		t.Fatalf("unexpected status %q", result.Status)
	}
}

// TestLookupOrderToolReturnsStructuredMiss keeps an unknown order as a structured successful
// lookup result with Found set to false.
func TestLookupOrderToolReturnsStructuredMiss(t *testing.T) {
	t.Parallel()

	response, callErr := newLookupOrderTool().Function(context.Background(), ai.ToolCall{
		ID:   "call_test",
		Type: "function",
		Name: "lookup_order",
		Args: json.RawMessage(`{"order_id":"lace-9999"}`),
	})
	if err := callErr; err != nil {
		t.Fatalf("lookup order: %v", err)
	}

	var result orderLookupResult
	if err := json.Unmarshal([]byte(response), &result); err != nil {
		t.Fatalf("decode result: %v", err)
	}
	if result.Found {
		t.Fatal("expected order not to be found")
	}
	if result.OrderID != "LACE-9999" {
		t.Fatalf("unexpected order ID %q", result.OrderID)
	}
}
