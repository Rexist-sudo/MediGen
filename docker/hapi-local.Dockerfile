FROM gcr.io/distroless/java21-debian13:nonroot

USER 65532:65532
WORKDIR /app

COPY --chown=nonroot:nonroot target/ROOT.war /app/main.war

ENTRYPOINT ["java", "--class-path", "/app/main.war", "-Dloader.path=main.war!/WEB-INF/classes/,main.war!/WEB-INF/,/app/extra-classes", "org.springframework.boot.loader.PropertiesLauncher"]
