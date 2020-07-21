function allocate_variables()
  io = IOBuffer()
  var = locvar("github_account")
  write(io, the_js_with_var)
  return String(take!(io))
end
