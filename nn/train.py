def train(model, loss_fn, X, y, epochs=100, lr=0.1):
    for epoch in range(epochs):
        # Forward
        pred = model.forward(X)
        loss = loss_fn.forward(pred, y)

        # Backward
        grad = loss_fn.backward()
        model.backward(grad)

        # Update
        model.step(lr)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
